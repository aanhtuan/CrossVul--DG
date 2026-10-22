"""DG-specific losses and updates, including the proposed CC-DRO objective.

References
----------
- IRMv1:     Arjovsky et al., 2019, eq. (1).
- GroupDRO:  Sagawa et al., 2020.
- Deep CORAL: Sun & Saenko, 2016.

CC-DRO (this work)
------------------
Cross-project SVD couples two shifts: a *domain* shift across projects and a
severe *class* imbalance. Vanilla GroupDRO takes the project as the group and
is class-blind: under ~1:19 imbalance the worst-case project risk can be driven
down by excelling on the (dominant) safe class, leaving rare-but-critical
vulnerability detection under-served and inconsistent across projects.

CC-DRO addresses this with two ingredients:

  (1) Class-conditional groups. Groups are (project x class) cells -> 2K cells
      for K source projects. GroupDRO is run over these cells so the worst case
      *includes* the "(project p, vulnerable)" cells explicitly.

  (2) Cross-project positive-class risk invariance. We add a penalty equal to
      the variance, across projects, of the positive-class (vulnerable) risk.
      Minimising it forces the model's vulnerability sensitivity to be
      consistent from one project to another -- the property cross-project SVD
      actually needs.

The full objective for a step is::

    L = sum_g q_g * R_g  +  lambda_inv * Var_p( R_{p, pos} )

where g ranges over the (project x class) cells present in the step, q is the
exponentiated-gradient DRO weight, and R_{p,pos} is project p's mean loss on its
positive samples.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weighted BCE (shared by ALL methods to handle imbalance fairly)
# ---------------------------------------------------------------------------

def bce_per_sample(logits: torch.Tensor, labels: torch.Tensor,
                   pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample BCE-with-logits (reduction='none').

    ``pos_weight`` (= N_neg / N_pos on the training fold) up-weights the rare
    positive (vulnerable) class so the optimiser does not collapse to the
    majority class. Class-blind baselines share this weight; CC-DRO defaults to unweighted within-cell BCE.
    """
    return F.binary_cross_entropy_with_logits(
        logits, labels.float(), pos_weight=pos_weight, reduction="none")


def bce_mean(logits: torch.Tensor, labels: torch.Tensor,
             pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return bce_per_sample(logits, labels, pos_weight).mean()


# ---------------------------------------------------------------------------
# IRM
# ---------------------------------------------------------------------------

def irm_penalty(logits: torch.Tensor, labels: torch.Tensor,
                pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """IRMv1 penalty: squared gradient of the risk w.r.t. a scalar dummy weight."""
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = bce_mean(logits * scale, labels, pos_weight)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).mean()


# ---------------------------------------------------------------------------
# GroupDRO (project-level groups) -- kept as a baseline
# ---------------------------------------------------------------------------

class GroupDROState:
    """Exponentiated-gradient state for GroupDRO over a fixed set of groups."""

    def __init__(self, n_groups: int, eta_q: float = 0.01,
                 init: str = "uniform", device: torch.device | str = "cpu"):
        if n_groups < 1 or eta_q < 0:
            raise ValueError("Invalid DRO group count or eta")
        self.n_groups = n_groups
        self.eta_q = eta_q
        self.device = torch.device(device)
        self.log_q = torch.full((n_groups,),-math.log(n_groups),device=self.device,dtype=torch.float64)

    @property
    def q(self):
        return self.log_q.exp().float()

    @q.setter
    def q(self,value):
        self.log_q=value.to(self.device,dtype=torch.float64).clamp_min(1e-300).log()
        self.log_q-=torch.logsumexp(self.log_q,dim=0)

    def update(self, group_losses: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            risks=group_losses.detach()
            if risks.shape!=self.log_q.shape or not torch.isfinite(risks).all():
                raise ValueError('Invalid/nonfinite group risks')
            self.log_q+=self.eta_q*risks.double()
            self.log_q-=torch.logsumexp(self.log_q,dim=0)
        return self.q

    def weighted_loss(self, group_losses: torch.Tensor) -> torch.Tensor:
        return (self.q * group_losses).sum()


# ---------------------------------------------------------------------------
# CC-DRO  (this work): class-conditional DRO + positive-class invariance
# ---------------------------------------------------------------------------

class CCDROState:
    """Exponentiated-gradient duals kept in log space, including checkpoint state.

    Missing cells keep their unnormalised historical mass. The stochastic
    objective renormalises duals over present cells, even if their global
    probabilities underflow. Initialize all source cells before training when
    population metadata is available.
    """
    def __init__(self,eta_q=.01,device='cpu'):
        if eta_q<0: raise ValueError('Negative dual learning rate')
        self.eta_q=eta_q;self.device=torch.device(device);self.log_q={}

    @property
    def q(self):
        return {k:v.exp().float() for k,v in self.log_q.items()}

    @q.setter
    def q(self,values):
        self.log_q={k:v.to(self.device,dtype=torch.float64).clamp_min(1e-300).log() for k,v in values.items()}
        if self.log_q:
            z=torch.logsumexp(torch.stack(list(self.log_q.values())),dim=0)
            self.log_q={k:v-z for k,v in self.log_q.items()}

    def _ensure_all(self,keys):
        prior=torch.logsumexp(torch.stack(list(self.log_q.values())),dim=0)-math.log(len(self.log_q)) if self.log_q else torch.tensor(0.,device=self.device,dtype=torch.float64)
        for key in sorted(keys):
            if key not in self.log_q: self.log_q[key]=prior.clone()

    def update(self,cell_losses):
        if not cell_losses: raise ValueError('No class-conditional cells')
        with torch.no_grad():
            self._ensure_all(cell_losses)
            for key,loss in cell_losses.items():
                if not torch.isfinite(loss).all(): raise ValueError('Nonfinite cell risk')
                self.log_q[key]+=self.eta_q*loss.detach().double()
            z=torch.logsumexp(torch.stack(list(self.log_q.values())),dim=0)
            self.log_q={k:v-z for k,v in self.log_q.items()}

    def weighted_loss(self,cell_losses):
        if not cell_losses: raise ValueError('No class-conditional cells')
        self._ensure_all(cell_losses)
        keys=list(cell_losses)
        weights=torch.softmax(torch.stack([self.log_q[k] for k in keys]),dim=0)
        return sum(weight.to(cell_losses[key].dtype)*cell_losses[key] for key,weight in zip(keys,weights))

    def weights_snapshot(self):
        return {f'p{p}_c{c}':float(value.detach().exp().cpu()) for (p,c),value in self.log_q.items()}


def class_conditional_cell_losses(
        per_sample_losses: torch.Tensor,
        labels: torch.Tensor,
        project_id: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Split one project's per-sample losses into its (project, 0) / (project, 1) cells.

    Returns a dict mapping ``(project_id, class)`` to the mean loss over the
    samples of that class present in the batch. Classes absent from the batch
    are simply omitted.
    """
    cells: Dict[Tuple[int, int], torch.Tensor] = {}
    for cls in (0, 1):
        mask = labels == cls
        if mask.any():
            cells[(project_id, cls)] = per_sample_losses[mask].mean()
    return cells


def positive_risk_invariance(pos_risks: List[torch.Tensor]) -> torch.Tensor:
    """Variance across projects of the positive-class risk.

    ``pos_risks`` is a list of scalar tensors, one per project that had >=1
    positive sample in the step. With <2 such projects the penalty is 0
    (variance undefined / not meaningful).
    """
    if len(pos_risks) < 2:
        return torch.zeros((), device=pos_risks[0].device) if pos_risks \
            else torch.zeros(())
    stacked = torch.stack(pos_risks)
    return stacked.var(unbiased=False)


# ---------------------------------------------------------------------------
# CORAL  (DA baseline)
# ---------------------------------------------------------------------------

def coral_loss(source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
    """Frobenius distance between source and target covariance matrices."""
    d = source_feats.size(1)

    def _cov(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        return (x.t() @ x) / max(x.size(0) - 1, 1)

    cs = _cov(source_feats)
    ct = _cov(target_feats)
    return ((cs - ct) ** 2).sum() / (4 * d * d)
