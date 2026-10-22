"""Resource-bounded trainers. Coupled DRO objectives use exact stochastic replay.

One optimizer step sees all environments at fixed parameters. A no-grad pass
computes detached cell risks and dual weights. A second pass, replaying identical
RNG/buffers, backpropagates the exact chain-rule coefficients one environment at
a time. No optimizer update occurs between environments.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import pooled_loader, sampling_budget
from .evaluate import compute_metrics, pr_auc
from .losses import (CCDROState, GroupDROState, bce_mean, bce_per_sample,
                     class_conditional_cell_losses, positive_risk_invariance,
                     irm_penalty, coral_loss)
from .models import DomainHead, grad_reverse


@dataclass
class TrainerConfig:
    method: str = 'erm'
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = .01
    warmup_steps: int = 0
    grad_clip: float = 1.
    early_stopping_patience: int = 3
    monitor: str = 'pr_auc'
    pos_weight: float = 1.
    irm_lambda_max: float = 10.
    irm_warmup_epochs: int = 1
    dro_eta_q: float = .01
    cc_eta_q: float = .01
    cc_lambda_inv: float = 1.
    cc_warmup_epochs: int = 1
    coral_weight: float = 1.
    dann_lambda: float = .1
    device: str = 'cpu'
    checkpoint_path: str | None = None
    resume: bool = True
    experiment_signature: str = ''
    amp: bool = False


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def _next(iters, loaders, i):
    try:
        return next(iters[i])
    except StopIteration:
        iters[i] = iter(loaders[i])
        return next(iters[i])


class _BaseTrainer:
    def __init__(self, model, cfg):
        if cfg.epochs < 1 or cfg.lr <= 0 or cfg.pos_weight <= 0 or cfg.grad_clip <= 0:
            raise ValueError('Invalid optimization configuration')
        self.cfg = cfg
        if cfg.amp and not cfg.device.startswith('cuda'):
            raise ValueError('Mixed precision requires CUDA in this implementation')
        self.model = model.to(cfg.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,foreach=False)
        self.scheduler = None
        self.pos_weight = torch.tensor(cfg.pos_weight, device=cfg.device)
        self.best_monitor = -float('inf')
        self.best_state = None
        self.patience_left = cfg.early_stopping_patience
        self.dro = None
        self.history = []
        self.identity = None
        self.scaler=torch.amp.GradScaler('cuda',enabled=cfg.amp,init_scale=1024.)
        self.optimizer_steps=0
        self.amp_skipped_steps=0

    def _logits(self,batch):
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=self.cfg.amp):
            return self.model(batch['input_ids'],batch['attention_mask']).float()

    def _backward(self,loss):
        self.scaler.scale(loss).backward()

    def _move(self, batch):
        return {k:v.to(self.cfg.device) if torch.is_tensor(v) else v for k,v in batch.items()}

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        ys,ps,gs = [],[],[]
        for batch in loader:
            batch = self._move(batch)
            pred = torch.sigmoid(self._logits(batch))
            ys.append(batch['label'].cpu().numpy()); ps.append(pred.cpu().numpy()); gs.append(batch['project_id'].cpu().numpy())
        if not ys:
            raise ValueError('Cannot evaluate an empty partition')
        return np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)

    def evaluate(self, loader, threshold=.5, project_ids=False):
        y,p,g = self.predict(loader)
        return compute_metrics(y,p,threshold,group=g if project_ids else None)

    def _extra_checkpoint_state(self):
        state = {}
        if self.dro is not None:
            state['q'] = self.dro.q
            state['log_q'] = self.dro.log_q
        if hasattr(self,'domain_head'):
            state['domain_head'] = self.domain_head.state_dict()
        return state

    def _load_extra_checkpoint_state(self, state):
        if self.dro is not None:
            q = state['q']
            logq=state['log_q']
            self.dro.log_q = {k:v.to(self.cfg.device) for k,v in logq.items()} if isinstance(logq,dict) else logq.to(self.cfg.device)
        if hasattr(self,'domain_head'):
            self.domain_head.load_state_dict(state['domain_head'])

    def _setup_scheduler(self, steps):
        total = steps*self.cfg.epochs
        warm = min(max(0,self.cfg.warmup_steps),max(0,total-1))
        self.schedule=dict(total_steps=total,requested_warmup_steps=self.cfg.warmup_steps,effective_warmup_steps=warm)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim,
            lambda step: (step+1)/warm if warm and step < warm else max(0.,(total-step)/max(1,total-warm)))

    def _identity(self, loaders, val_loader):
        config = asdict(self.cfg)
        for k in ('checkpoint_path','resume','device'):
            config.pop(k)
        def fingerprint(ds):
            if hasattr(ds,'datasets'):
                return [fingerprint(x) for x in ds.datasets]
            if hasattr(ds,'df'):
                frame = ds.df.sort_index(axis=1)
                digest=hashlib.sha256()
                for start in range(0,len(frame),1000):
                    digest.update(frame.iloc[start:start+1000].to_json(date_format='iso',orient='split').encode())
                return dict(rows=digest.hexdigest(),
                            max_length=getattr(ds,'max_length',None), tokenizer=type(getattr(ds,'tokenizer',None)).__qualname__)
            if self.cfg.checkpoint_path and not self.cfg.experiment_signature:
                raise ValueError('Checkpoint requires dataset identity for this dataset type')
            return dict(type=type(ds).__qualname__, n=len(ds))
        model = dict(type=type(self.model).__qualname__, backbone=getattr(self.model,'model_name',None),
                     architecture=repr(self.model), shapes={k:list(v.shape) for k,v in self.model.state_dict().items()})
        identity = dict(version='training-v3-exact-replay-logduals',model=model,config=config,
                        sources=[fingerprint(l.dataset) for l in loaders],validation=fingerprint(val_loader.dataset),
                        target=fingerprint(self.target_loader.dataset) if hasattr(self,'target_loader') else None,
                        sampler=[dict(batch_size=l.batch_size or getattr(l.batch_sampler,'batch_size',None),
                                      balanced_seed=getattr(l.batch_sampler,'seed',None), batch_sampler=type(l.batch_sampler).__qualname__,
                                      sampler=type(l.sampler).__qualname__) for l in loaders])
        return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def _save_checkpoint(self, epoch):
        if not self.cfg.checkpoint_path:
            return
        path = Path(self.cfg.checkpoint_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        sampler_state = [dict(epoch=getattr(l.batch_sampler,'epoch',None),
                              generator=l.generator.get_state() if l.generator else None,
                              sampler_generator=l.sampler.generator.get_state() if getattr(l.sampler,'generator',None) else None)
                         for l in self._stateful_loaders]
        state = dict(version=3,identity=self.identity,completed_epoch=epoch,model_state_dict=self.model.state_dict(),
                     optimizer_state_dict=self.optim.state_dict(),scheduler_state_dict=self.scheduler.state_dict(),
                     best_monitor=self.best_monitor,best_state=self.best_state,patience_left=self.patience_left,
                     method=self.cfg.method,extra_state=self._extra_checkpoint_state(),rng=capture_rng(),samplers=sampler_state,
                     history=self.history,config=asdict(self.cfg),scaler=self.scaler.state_dict(),
                     optimizer_steps=self.optimizer_steps,amp_skipped_steps=self.amp_skipped_steps)
        temp = Path(str(path)+'.tmp'); torch.save(state,temp); os.replace(temp,path)
        # Explicit standalone best-validation artifact; never used as resume state.
        best = path.with_name(path.stem+'.best.pt')
        temp = Path(str(best)+'.tmp')
        torch.save(dict(identity=self.identity,model_state_dict=self.best_state,best_monitor=self.best_monitor),temp)
        os.replace(temp,best)

    def _resume_if_available(self):
        path = Path(self.cfg.checkpoint_path) if self.cfg.checkpoint_path else None
        if not self.cfg.resume or path is None or not path.exists():
            return 0
        state = torch.load(path,map_location='cpu',weights_only=False)
        if state.get('version') != 3 or state.get('identity') != self.identity:
            raise ValueError('Checkpoint experiment identity mismatch (data/config/backbone/objective)')
        self.model.load_state_dict(state['model_state_dict']); self.optim.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.best_monitor=state['best_monitor']; self.best_state=state['best_state']; self.patience_left=state['patience_left']
        self.history=state['history']; self._load_extra_checkpoint_state(state['extra_state'])
        self.scaler.load_state_dict(state.get('scaler',{}))
        self.optimizer_steps=state.get('optimizer_steps',0)
        self.amp_skipped_steps=state.get('amp_skipped_steps',0)
        for loader,saved in zip(self._stateful_loaders,state['samplers']):
            if saved['epoch'] is not None:
                loader.batch_sampler.epoch = saved['epoch']
            if saved['generator'] is not None:
                loader.generator.set_state(saved['generator'])
            if saved['sampler_generator'] is not None:
                loader.sampler.generator.set_state(saved['sampler_generator'])
        restore_rng(state['rng'])
        return state['completed_epoch']+1

    def load_best(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    @staticmethod
    def _balanced_environment_steps(loaders):
        if not loaders or any(len(l)==0 for l in loaders):
            raise ValueError('No usable source environments')
        return max(1,int(np.ceil(sum(map(len,loaders))/len(loaders))))

    def _optim_step(self):
        params = [p for group in self.optim.param_groups for p in group['params']]
        self.scaler.unscale_(self.optim)
        norm = nn.utils.clip_grad_norm_(params,self.cfg.grad_clip,error_if_nonfinite=not self.cfg.amp)
        scale=self.scaler.get_scale()
        self.scaler.step(self.optim);self.scaler.update()
        if self.scaler.get_scale()>=scale:
            self.scheduler.step();self.optimizer_steps+=1
        else:
            self.amp_skipped_steps+=1
        return float(norm)

    def _coupled_step(self, batches, lam):
        """Return detached objective diagnostics and accumulate exact gradients."""
        records, cells = [], {}
        for i, raw in enumerate(batches):
            state = capture_rng()
            buffers = {k:v.detach().clone() for k,v in self.model.named_buffers()}
            batch = self._move(raw)
            with torch.no_grad():
                logits = self._logits(batch)
                values = (class_conditional_cell_losses(bce_per_sample(logits,batch['label'],self.pos_weight),batch['label'],i)
                          if self.cfg.method=='cc-dro' else {(i,0):bce_mean(logits,batch['label'],self.pos_weight)})
            cells.update({k:v.detach().requires_grad_() for k,v in values.items()})
            records.append((raw,state,buffers,list(values)))
        end_rng = capture_rng()
        if self.cfg.method=='cc-dro':
            self.dro.update(cells)
            weighted = self.dro.weighted_loss(cells)
            penalty = positive_risk_invariance([v for (p,c),v in cells.items() if c==1]).to(self.cfg.device)
        else:
            risks = torch.stack(list(cells.values()))
            self.dro.update(risks)
            weighted = self.dro.weighted_loss(risks)
            penalty = weighted.new_zeros(())
        objective = weighted + lam*penalty
        classification_coeff=torch.autograd.grad(weighted,tuple(cells.values()),retain_graph=True)
        penalty_coeff=torch.autograd.grad(lam*penalty,tuple(cells.values()),retain_graph=True,allow_unused=True) if penalty.requires_grad else ()
        coefficients = torch.autograd.grad(objective,tuple(cells.values()))
        coeff = dict(zip(cells,coefficients))
        # Replay one environment graph at a time; coefficients include the variance derivative.
        for i,(raw,state,buffers,keys) in enumerate(records):
            restore_rng(state)
            for name,buf in self.model.named_buffers():
                buf.copy_(buffers[name])
            batch=self._move(raw)
            logits=self._logits(batch)
            values=(class_conditional_cell_losses(bce_per_sample(logits,batch['label'],self.pos_weight),batch['label'],i)
                    if self.cfg.method=='cc-dro' else {(i,0):bce_mean(logits,batch['label'],self.pos_weight)})
            self._backward(sum(coeff[k]*values[k] for k in keys))
        restore_rng(end_rng)
        q = list(self.dro.q.values()) if isinstance(self.dro.q,dict) else list(self.dro.q)
        return dict(classification_loss=float(weighted.detach()),invariance_penalty=float(penalty.detach()),
                    weighted_penalty=float((lam*penalty).detach()),loss=float(objective.detach()),
                    penalty_to_classification_ratio=float((lam*penalty).detach())/max(float(weighted.detach()),1e-12),
                    classification_coefficient_l2=float(torch.stack(classification_coeff).norm()),
                    invariance_coefficient_l2=float(torch.stack([v for v in penalty_coeff if v is not None]).norm()) if any(v is not None for v in penalty_coeff) else 0.,
                    present_cells=len(cells),total_cells=len(q),
                    q_min=min(float(v) for v in q),q_max=max(float(v) for v in q),
                    q_entropy=-sum(float(v)*np.log(max(float(v),1e-30)) for v in q))

    def fit(self, train_loaders, val_loader):
        method=self.cfg.method
        loaders = [pooled_loader(train_loaders)] if method in ('erm','da-coral','da-dann') else train_loaders
        self._loaders=loaders
        self._stateful_loaders=loaders+[val_loader]+([self.target_loader] if hasattr(self,'target_loader') else [])
        n_steps = len(loaders[0]) if method in ('erm','da-coral','da-dann') else self._balanced_environment_steps(loaders)
        self.identity=self._identity(loaders,val_loader)
        if method=='cc-dro':
            self.dro=CCDROState(self.cfg.cc_eta_q,self.cfg.device)
            keys=[(i,int(c)) for i,l in enumerate(loaders) if hasattr(l.dataset,'df') for c in sorted(l.dataset.df.label.unique())]
            if keys:
                self.dro.q={k:torch.tensor(1./len(keys),device=self.cfg.device) for k in keys}
        elif method in ('dro','pd-dro'):
            self.dro=GroupDROState(len(loaders),self.cfg.dro_eta_q,device=self.cfg.device)
        self._setup_scheduler(n_steps)
        start=self._resume_if_available()
        self.sampling=sampling_budget(loaders,n_steps)
        for epoch in range(start,self.cfg.epochs):
            if self.patience_left<=0:
                break
            self.model.train()
            if hasattr(self,'domain_head'):
                self.domain_head.train()
            iters=[iter(l) for l in loaders]
            target_iter=[iter(self.target_loader)] if hasattr(self,'target_loader') else None
            logs=[]
            exposures={}
            def observe(raw):
                for project,index,label in zip(raw['project_id'].tolist(),raw.get('sample_index',torch.arange(len(raw['label']))).tolist(),raw['label'].tolist()):
                    entry=exposures.setdefault(project,dict(seen=set(),draws=0,positive=0))
                    entry['seen'].add(index);entry['draws']+=1;entry['positive']+=int(label)
            lam=self.cfg.cc_lambda_inv if method=='cc-dro' and epoch>=self.cfg.cc_warmup_epochs else 0.
            for _ in range(n_steps):
                self.optim.zero_grad(set_to_none=True)
                if self.dro is not None:
                    # Hold only current CPU batches plus one environment's autograd graph.
                    batches=[_next(iters,loaders,i) for i in range(len(loaders))]
                    for raw in batches: observe(raw)
                    logs.append(self._coupled_step(batches,lam))
                else:
                    total=0.
                    for i in range(len(loaders)):
                        raw=_next(iters,loaders,i)
                        observe(raw)
                        batch=self._move(raw)
                        if method in ('da-coral','da-dann'):
                            target=self._move(_next(target_iter,[self.target_loader],0))
                            src=self.model.encode(batch['input_ids'],batch['attention_mask'])
                            tgt=self.model.encode(target['input_ids'],target['attention_mask'])
                            logits=self.model.classifier(src).squeeze(-1)
                            loss=bce_mean(logits,batch['label'],self.pos_weight)
                            if method=='da-coral':
                                loss=loss+self.cfg.coral_weight*coral_loss(src,tgt)
                            else:
                                domain_y=torch.cat([torch.zeros(len(src),device=self.cfg.device,dtype=torch.long),torch.ones(len(tgt),device=self.cfg.device,dtype=torch.long)])
                                loss=loss+nn.functional.cross_entropy(self.domain_head(grad_reverse(torch.cat([src,tgt]),self.cfg.dann_lambda)),domain_y)
                        else:
                            logits=self._logits(batch)
                            loss=bce_mean(logits,batch['label'],self.pos_weight)
                            if method=='irm':
                                irm_lam=self.cfg.irm_lambda_max if epoch>=self.cfg.irm_warmup_epochs else 0.
                                loss=(loss+irm_lam*irm_penalty(logits,batch['label'],self.pos_weight))/max(1.,irm_lam)
                        loss=loss/len(loaders)
                        self._backward(loss); total+=float(loss.detach())
                    logs.append(dict(loss=total))
                logs[-1]['gradient_norm']=self._optim_step()
            val=self._val_monitor(val_loader)
            record={k:float(np.mean([x[k] for x in logs])) for k in logs[0]}
            record.update(epoch=epoch,val_monitor=val,lambda_inv=lam)
            record.update(optimizer_steps=self.optimizer_steps,amp_skipped_steps=self.amp_skipped_steps,
                          amp_scale=self.scaler.get_scale())
            record['sampling_effective']=[dict(project_id=p,draws=e['draws'],unique_samples=len(e['seen']),
                repeated_draws=e['draws']-len(e['seen']),positive_draws=e['positive'],negative_draws=e['draws']-e['positive'])
                for p,e in sorted(exposures.items())]
            self.history.append(record)
            print(f'[{method}] '+json.dumps(record,sort_keys=True))
            if val>self.best_monitor:
                self.best_monitor=val
                self.best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                self.patience_left=self.cfg.early_stopping_patience
            else:
                self.patience_left-=1
            self._save_checkpoint(epoch)
        self.load_best()
        return self

    def _val_monitor(self,loader):
        y,p,_=self.predict(loader)
        value=pr_auc(y,p) if self.cfg.monitor=='pr_auc' else compute_metrics(y,p)[self.cfg.monitor]
        if not np.isfinite(value):
            raise ValueError('Validation selection metric undefined; fold cannot select a model')
        return float(value)


class ERMTrainer(_BaseTrainer): pass
class IRMTrainer(_BaseTrainer): pass
class GroupDROTrainer(_BaseTrainer): pass
class CCDROTrainer(_BaseTrainer): pass


class CORALTrainer(_BaseTrainer):
    def __init__(self,model,cfg,target_loader):
        super().__init__(model,cfg)
        if target_loader is None or len(target_loader)==0:
            raise ValueError('DA requires a nonempty disjoint unlabelled target loader')
        self.target_loader=target_loader


class DANNTrainer(CORALTrainer):
    def __init__(self,model,cfg,target_loader,n_source_domains=1):
        super().__init__(model,cfg,target_loader)
        first=next(m for m in model.classifier.modules() if isinstance(m,nn.Linear))
        self.domain_head=DomainHead(first.in_features,2).to(cfg.device)
        self.optim.add_param_group({'params':self.domain_head.parameters()})


def build_trainer(method,model,cfg,target_loader=None,n_source_domains=1):
    if method != cfg.method:
        raise ValueError('Method/config mismatch')
    classes={'erm':ERMTrainer,'irm':IRMTrainer,'dro':GroupDROTrainer,'cc-dro':CCDROTrainer,'pd-dro':GroupDROTrainer}
    if method in classes:
        return classes[method](model,cfg)
    if method=='da-coral': return CORALTrainer(model,cfg,target_loader)
    if method=='da-dann': return DANNTrainer(model,cfg,target_loader,n_source_domains)
    raise ValueError(f'Unknown method {method}')
