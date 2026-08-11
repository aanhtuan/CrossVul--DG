"""Training loops for ERM, IRM, GroupDRO, the proposed CC-DRO, and DA baselines.

All trainers share ``.fit(train_loaders, val_loader)`` where ``train_loaders``
is a list of per-environment (per-project) DataLoaders for DG methods, or a
length-1 list for ERM. Key properties shared by every method for a *fair*
imbalanced comparison:

  * BCE is weighted by ``pos_weight`` (= N_neg / N_pos on the training fold).
  * Model selection (early stopping) monitors validation **PR-AUC**, which is
    threshold-free and robust under imbalance (val-F1 at 0.5 is not).
  * The decision threshold itself is tuned *afterwards* on val (see run_lopo).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import compute_metrics, pr_auc
from .losses import (
    CCDROState, GroupDROState, bce_mean, bce_per_sample,
    class_conditional_cell_losses, coral_loss, irm_penalty,
    positive_risk_invariance,
)
from .models import DomainHead, EncoderClassifier, grad_reverse


@dataclass
class TrainerConfig:
    method: str = "erm"            # erm / irm / dro / cc-dro / pd-dro / da-coral / da-dann
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    early_stopping_patience: int = 3
    monitor: str = "pr_auc"        # val metric for model selection
    pos_weight: float = 1.0        # N_neg / N_pos on the training fold
    # IRM
    irm_lambda_max: float = 10.0
    irm_warmup_epochs: int = 1
    # GroupDRO / CC-DRO
    dro_eta_q: float = 0.01
    # CC-DRO invariance penalty
    cc_lambda_inv: float = 1.0
    cc_warmup_epochs: int = 1
    # DA
    coral_weight: float = 1.0
    dann_lambda: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Checkpoint/resume
    checkpoint_path: Optional[str] = None
    resume: bool = True


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _BaseTrainer:
    """Shared optimisation / model-selection plumbing."""

    def __init__(self, model: nn.Module, cfg: TrainerConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.optim = torch.optim.AdamW(self.model.parameters(),
                                       lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.pos_weight = torch.tensor(float(cfg.pos_weight), device=cfg.device)
        self.best_monitor = -float("inf")
        self.best_state = None
        self.patience_left = cfg.early_stopping_patience

    def _extra_checkpoint_state(self) -> dict:
        """Trạng thái riêng của từng phương pháp."""
        return {}

    def _load_extra_checkpoint_state(self, state: dict) -> None:
        """Khôi phục trạng thái riêng của từng phương pháp."""
        pass

    def _save_checkpoint(self, completed_epoch: int) -> None:
        if not self.cfg.checkpoint_path:
            return

        path = Path(self.cfg.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "completed_epoch": completed_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "best_monitor": self.best_monitor,
            "best_state": self.best_state,
            "patience_left": self.patience_left,
            "method": self.cfg.method,
            "extra_state": self._extra_checkpoint_state(),
        }

        # Lưu tạm rồi đổi tên để tránh checkpoint hỏng
        # nếu Colab ngắt đúng lúc đang ghi file.
        temporary_path = Path(str(path) + ".tmp")
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)

        print(
            f"Đã lưu checkpoint sau epoch {completed_epoch + 1}: {path}"
        )

    def _resume_if_available(self) -> int:
        if not self.cfg.resume or not self.cfg.checkpoint_path:
            return 0

        path = Path(self.cfg.checkpoint_path)

        if not path.exists():
            print(f"Chưa có checkpoint, bắt đầu từ epoch 1: {path}")
            return 0

        checkpoint = torch.load(
            path,
            map_location=self.cfg.device,
            weights_only=False,
        )

        saved_method = checkpoint.get("method")

        if saved_method and saved_method != self.cfg.method:
            raise ValueError(
                f"Checkpoint thuộc method={saved_method!r}, "
                f"nhưng hiện tại đang chạy method={self.cfg.method!r}"
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_monitor = checkpoint.get(
            "best_monitor",
            -float("inf"),
        )
        self.best_state = checkpoint.get("best_state")
        self.patience_left = checkpoint.get(
            "patience_left",
            self.cfg.early_stopping_patience,
        )

        self._load_extra_checkpoint_state(
            checkpoint.get("extra_state", {})
        )

        start_epoch = int(checkpoint["completed_epoch"]) + 1

        print(f"Đã tìm thấy checkpoint: {path}")
        print(f"Đã hoàn thành epoch {start_epoch}")
        print(f"Tiếp tục từ epoch {start_epoch + 1}")

        return start_epoch

    # ------------------------------------------------------------------
    def _move(self, batch: dict) -> dict:
        return {k: (v.to(self.cfg.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_true, y_prob, project_id) over a loader."""
        self.model.eval()
        ys, ps, gs = [], [], []
        for batch in loader:
            batch = self._move(batch)
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(batch["label"].cpu().numpy())
            gs.append(batch["project_id"].cpu().numpy())
        return (np.concatenate(ys), np.concatenate(ps), np.concatenate(gs))

    def evaluate(self, loader: DataLoader, threshold: float = 0.5,
                 project_ids: bool = False) -> dict:
        y, p, g = self.predict(loader)
        m = compute_metrics(y, p, threshold=threshold,
                            group=g if project_ids else None)
        return m

    def _val_monitor(self, val_loader: DataLoader) -> float:
        y, p, _ = self.predict(val_loader)
        if self.cfg.monitor == "pr_auc":
            v = pr_auc(y, p)
        else:
            v = compute_metrics(y, p, threshold=0.5)[self.cfg.monitor]
        return float(v) if v == v else -1.0    # NaN -> -1

    def _maybe_save_best(self, val_value: float) -> bool:
        if val_value > self.best_monitor:
            self.best_monitor = val_value
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in self.model.state_dict().items()}
            self.patience_left = self.cfg.early_stopping_patience
            return True
        self.patience_left -= 1
        return False

    def load_best(self) -> None:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# ERM
# ---------------------------------------------------------------------------

class ERMTrainer(_BaseTrainer):
    def fit(self, train_loaders: List[DataLoader], val_loader: DataLoader):
        for epoch in range(self.cfg.epochs):
            self.model.train()
            total, count = 0.0, 0
            for loader in train_loaders:
                for batch in tqdm(loader, desc=f"ERM epoch {epoch+1}", leave=False):
                    batch = self._move(batch)
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    loss = bce_mean(logits, batch["label"], self.pos_weight)
                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optim.step()
                    total += loss.item() * batch["label"].size(0)
                    count += batch["label"].size(0)
            v = self._val_monitor(val_loader)
            print(f"[ERM] epoch={epoch+1} train_loss={total/max(count,1):.4f} "
                  f"val_{self.cfg.monitor}={v:.4f}")
            self._maybe_save_best(v)
            if self.patience_left <= 0:
                print("Early stopping."); break
        self.load_best()


# ---------------------------------------------------------------------------
# IRM
# ---------------------------------------------------------------------------

class IRMTrainer(_BaseTrainer):
    def fit(self, train_loaders: List[DataLoader], val_loader: DataLoader):
        for epoch in range(self.cfg.epochs):
            self.model.train()
            iters = [iter(l) for l in train_loaders]
            n_steps = min(len(l) for l in train_loaders)
            lam = self.cfg.irm_lambda_max if epoch >= self.cfg.irm_warmup_epochs else 0.0
            tot = 0.0
            for _ in tqdm(range(n_steps), desc=f"IRM epoch {epoch+1}", leave=False):
                erm_terms, pen_terms = [], []
                for ei, it in enumerate(iters):
                    try:
                        batch = next(it)
                    except StopIteration:
                        iters[ei] = iter(train_loaders[ei]); batch = next(iters[ei])
                    batch = self._move(batch)
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    erm_terms.append(bce_mean(logits, batch["label"], self.pos_weight))
                    pen_terms.append(irm_penalty(logits, batch["label"], self.pos_weight))
                loss = sum(erm_terms) / len(erm_terms) + lam * (sum(pen_terms) / len(pen_terms))
                if lam > 1.0:
                    loss = loss / lam
                self.optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optim.step(); tot += loss.item()
            v = self._val_monitor(val_loader)
            print(f"[IRM] epoch={epoch+1} lam={lam:.1f} loss={tot/max(n_steps,1):.4f} "
                  f"val_{self.cfg.monitor}={v:.4f}")
            self._maybe_save_best(v)
            if self.patience_left <= 0:
                print("Early stopping."); break
        self.load_best()


# ---------------------------------------------------------------------------
# GroupDRO  (project-level groups)
# ---------------------------------------------------------------------------

class GroupDROTrainer(_BaseTrainer):
    def fit(self, train_loaders: List[DataLoader], val_loader: DataLoader):
        dro = GroupDROState(len(train_loaders), eta_q=self.cfg.dro_eta_q,
                            device=self.cfg.device)
        for epoch in range(self.cfg.epochs):
            self.model.train()
            iters = [iter(l) for l in train_loaders]
            n_steps = min(len(l) for l in train_loaders)
            tot = 0.0
            for _ in tqdm(range(n_steps), desc=f"DRO epoch {epoch+1}", leave=False):
                losses = []
                for ei, it in enumerate(iters):
                    try:
                        batch = next(it)
                    except StopIteration:
                        iters[ei] = iter(train_loaders[ei]); batch = next(iters[ei])
                    batch = self._move(batch)
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    losses.append(bce_mean(logits, batch["label"], self.pos_weight))
                losses_t = torch.stack(losses)
                dro.update(losses_t)
                weighted = dro.weighted_loss(losses_t)
                self.optim.zero_grad(); weighted.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optim.step(); tot += weighted.item()
            v = self._val_monitor(val_loader)
            print(f"[DRO] epoch={epoch+1} loss={tot/max(n_steps,1):.4f} "
                  f"val_{self.cfg.monitor}={v:.4f}")
            self._maybe_save_best(v)
            if self.patience_left <= 0:
                print("Early stopping."); break
        self.load_best()


# ---------------------------------------------------------------------------
# CC-DRO  (this work): class-conditional DRO + positive-class invariance
# ---------------------------------------------------------------------------

class CCDROTrainer(_BaseTrainer):
    class CCDROTrainer(_BaseTrainer):
        """GroupDRO over (project x class) cells plus a cross-project positive-risk
    invariance penalty. See ``losses`` for the objective."""

    def __init__(self, model: nn.Module, cfg: TrainerConfig):
        super().__init__(model, cfg)
        self.dro = None

    def _extra_checkpoint_state(self) -> dict:
        if self.dro is None:
            return {}

        return {
            "ccdro_q": {
                key: value.detach().cpu()
                for key, value in self.dro.q.items()
            }
        }

    def _load_extra_checkpoint_state(self, state: dict) -> None:
        if "ccdro_q" in state:
            self.dro.q = {
                key: value.to(self.cfg.device)
                for key, value in state["ccdro_q"].items()
            }

    def fit(
        self,
        train_loaders: List[DataLoader],
        val_loader: DataLoader,
    ):
        self.dro = CCDROState(
            eta_q=self.cfg.dro_eta_q,
            device=self.cfg.device,
        )

        start_epoch = self._resume_if_available()

        for epoch in range(start_epoch, self.cfg.epochs):
            self.model.train()

            iters = [iter(loader) for loader in train_loaders]
            n_steps = min(len(loader) for loader in train_loaders)

            lam = (
                self.cfg.cc_lambda_inv
                if epoch >= self.cfg.cc_warmup_epochs
                else 0.0
            )

            tot = 0.0

            for _ in tqdm(
                range(n_steps),
                desc=f"CC-DRO epoch {epoch + 1}",
                leave=False,
            ):
                cell_losses: Dict[
                    Tuple[int, int],
                    torch.Tensor,
                ] = {}

                pos_risks: List[torch.Tensor] = []

                for ei, iterator in enumerate(iters):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iters[ei] = iter(train_loaders[ei])
                        batch = next(iters[ei])

                    batch = self._move(batch)

                    logits = self.model(
                        batch["input_ids"],
                        batch["attention_mask"],
                    )

                    per_sample = bce_per_sample(
                        logits,
                        batch["label"],
                        self.pos_weight,
                    )

                    cells = class_conditional_cell_losses(
                        per_sample,
                        batch["label"],
                        ei,
                    )

                    cell_losses.update(cells)

                    if (ei, 1) in cells:
                        pos_risks.append(cells[(ei, 1)])

                self.dro.update(cell_losses)
                weighted = self.dro.weighted_loss(cell_losses)
                penalty = positive_risk_invariance(pos_risks)
                loss = weighted + lam * penalty

                self.optim.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.grad_clip,
                )

                self.optim.step()
                tot += loss.item()

            v = self._val_monitor(val_loader)

            print(
                f"[CC-DRO] epoch={epoch + 1} "
                f"lam_inv={lam:.2f} "
                f"loss={tot / max(n_steps, 1):.4f} "
                f"val_{self.cfg.monitor}={v:.4f}"
            )

            self._maybe_save_best(v)

            # Lưu sau khi hoàn thành trọn vẹn epoch hiện tại.
            self._save_checkpoint(epoch)

            if self.patience_left <= 0:
                print("Early stopping.")
                break

        self.load_best()

# ---------------------------------------------------------------------------
# DA-CORAL  (uses unlabelled target features, NOT strict DG)
# ---------------------------------------------------------------------------

class CORALTrainer(_BaseTrainer):
    def __init__(self, model: EncoderClassifier, cfg: TrainerConfig,
                 target_loader: DataLoader):
        super().__init__(model, cfg)
        self.target_loader = target_loader

    def fit(self, train_loaders: List[DataLoader], val_loader: DataLoader):
        for epoch in range(self.cfg.epochs):
            self.model.train()
            tgt_it = iter(self.target_loader); tot = 0.0; steps = 0
            for loader in train_loaders:
                for batch in tqdm(loader, desc=f"CORAL epoch {epoch+1}", leave=False):
                    batch = self._move(batch)
                    try:
                        tgt = next(tgt_it)
                    except StopIteration:
                        tgt_it = iter(self.target_loader); tgt = next(tgt_it)
                    tgt = self._move(tgt)
                    src_feat = self.model.encode(batch["input_ids"], batch["attention_mask"])
                    tgt_feat = self.model.encode(tgt["input_ids"], tgt["attention_mask"])
                    logits = self.model.classifier(src_feat).squeeze(-1)
                    cls = bce_mean(logits, batch["label"], self.pos_weight)
                    loss = cls + self.cfg.coral_weight * coral_loss(src_feat, tgt_feat)
                    self.optim.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optim.step(); tot += loss.item(); steps += 1
            v = self._val_monitor(val_loader)
            print(f"[CORAL] epoch={epoch+1} loss={tot/max(steps,1):.4f} "
                  f"val_{self.cfg.monitor}={v:.4f}")
            self._maybe_save_best(v)
            if self.patience_left <= 0:
                break
        self.load_best()


# ---------------------------------------------------------------------------
# DA-DANN  (adversarial domain classifier; unlabelled target = held-out project)
# ---------------------------------------------------------------------------

class DANNTrainer(_BaseTrainer):
    def __init__(self, model: EncoderClassifier, cfg: TrainerConfig,
                 target_loader: DataLoader, n_source_domains: int = 1):
        super().__init__(model, cfg)
        self.domain_head = DomainHead(in_dim=768, n_domains=2).to(cfg.device)
        self.optim.add_param_group({"params": self.domain_head.parameters()})
        self.target_loader = target_loader

    def fit(self, train_loaders: List[DataLoader], val_loader: DataLoader):
        for epoch in range(self.cfg.epochs):
            self.model.train(); self.domain_head.train()
            tgt_it = iter(self.target_loader); tot = 0.0; steps = 0
            for loader in train_loaders:
                for batch in tqdm(loader, desc=f"DANN epoch {epoch+1}", leave=False):
                    batch = self._move(batch)
                    try:
                        tgt = next(tgt_it)
                    except StopIteration:
                        tgt_it = iter(self.target_loader); tgt = next(tgt_it)
                    tgt = self._move(tgt)
                    src_feat = self.model.encode(batch["input_ids"], batch["attention_mask"])
                    tgt_feat = self.model.encode(tgt["input_ids"], tgt["attention_mask"])
                    logits = self.model.classifier(src_feat).squeeze(-1)
                    cls_loss = bce_mean(logits, batch["label"], self.pos_weight)
                    feats = torch.cat([src_feat, tgt_feat], dim=0)
                    dom_labels = torch.cat([
                        torch.zeros(src_feat.size(0), dtype=torch.long, device=self.cfg.device),
                        torch.ones(tgt_feat.size(0), dtype=torch.long, device=self.cfg.device),
                    ])
                    dom_logits = self.domain_head(grad_reverse(feats, self.cfg.dann_lambda))
                    dom_loss = nn.functional.cross_entropy(dom_logits, dom_labels)
                    loss = cls_loss + dom_loss
                    self.optim.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.domain_head.parameters()),
                        self.cfg.grad_clip)
                    self.optim.step(); tot += loss.item(); steps += 1
            v = self._val_monitor(val_loader)
            print(f"[DANN] epoch={epoch+1} loss={tot/max(steps,1):.4f} "
                  f"val_{self.cfg.monitor}={v:.4f}")
            self._maybe_save_best(v)
            if self.patience_left <= 0:
                break
        self.load_best()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_trainer(method: str, model: EncoderClassifier, cfg: TrainerConfig,
                  target_loader: Optional[DataLoader] = None,
                  n_source_domains: int = 1) -> _BaseTrainer:
    method = method.lower()
    if method == "erm":
        return ERMTrainer(model, cfg)
    if method == "irm":
        return IRMTrainer(model, cfg)
    if method in {"dro", "pd-dro"}:
        return GroupDROTrainer(model, cfg)
    if method == "cc-dro":
        return CCDROTrainer(model, cfg)
    if method == "da-coral":
        if target_loader is None:
            raise ValueError("CORAL requires target_loader")
        return CORALTrainer(model, cfg, target_loader)
    if method == "da-dann":
        if target_loader is None:
            raise ValueError("DANN requires target_loader")
        return DANNTrainer(model, cfg, target_loader, n_source_domains)
    raise ValueError(f"Unknown method {method!r}")
