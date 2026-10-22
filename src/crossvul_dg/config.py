"""Validate all consumed configuration fields before model/data allocation."""
import math
import re


def validate(cfg):
    errors=[]
    if not isinstance(cfg,dict): return ['Configuration root must be a mapping']
    specs={
      'encoder.name':str,'encoder.revision':str,'encoder.max_length':(int,1,512),'encoder.hidden_size':(int,1,None),
      'encoder.dropout':(float,0,1),'train.batch_size':(int,1,None),'eval.batch_size':(int,1,None),
      'train.epochs':(int,1,None),'train.lr':(float,1e-15,None),'train.weight_decay':(float,0,None),
      'train.warmup_steps':(int,0,None),'train.grad_clip':(float,1e-15,None),
      'train.early_stopping_patience':(int,1,None),'train.num_workers':(int,0,2),
      'train.resume':bool,'train.device':str,'train.monitor':str,
      'irm.lambda_max':(float,0,None),'irm.warmup_epochs':(int,0,None),
      'dro.eta_q':(float,0,None),'cc_dro.eta_q':(float,0,None),'cc_dro.lambda_inv':(float,0,None),
      'cc_dro.warmup_epochs':(int,0,None),'cc_dro.balanced_batches':bool,'cc_dro.use_pos_weight':bool,
      'da.coral_weight':(float,0,None),'da.dann_lambda':(float,0,None),'da.target_ratio':(float,1e-15,1-1e-15),
      'pseudo.n_clusters':(int,1,None),'pseudo.embedding_subset':(int,1,None),
      'data.top_k':(int,1,None),'data.train_ratio':(float,1e-15,1),'data.val_ratio':(float,1e-15,1),
      'data.test_ratio':(float,1e-15,1),'data.dedup':bool}
    def get(key):
        node=cfg
        for part in key.split('.'):
            node=node.get(part) if isinstance(node,dict) else None
        return node
    for key,spec in specs.items():
        value=get(key)
        if isinstance(spec,tuple):
            typ,low,high=spec
            valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
            valid=valid and (typ!=int or isinstance(value,int))
            valid=valid and value>=low and (high is None or value<=high)
        else:
            valid=isinstance(value,spec) and (spec!=str or bool(value))
        if not valid: errors.append(f'Invalid/missing {key}: {value!r}')
    revision=get('encoder.revision')
    if isinstance(revision,str) and not re.fullmatch('[0-9a-f]{40}',revision):
        errors.append('encoder.revision must be an immutable 40-character Hub commit SHA')
    if get('train.device') not in ('cpu','cuda'):
        errors.append('train.device must be cpu or cuda')
    if get('train.monitor') not in ('pr_auc','roc_auc','f1','precision','recall','accuracy'):
        errors.append('Unknown train.monitor')
    if get('cc_dro.balanced_batches') and isinstance(get('train.batch_size'),int) and get('train.batch_size')<2:
        errors.append('Class-balanced CC-DRO requires batch_size >= 2')
    if not errors:
        if abs(sum(get('data.'+k+'_ratio') for k in ('train','val','test'))-1)>1e-8:
            errors.append('Data split ratios must sum to one')
    if get('data.dedup') is not True:
        errors.append('Protocol v2 requires data.dedup=true')
    for key in ('train.amp','encoder.gradient_checkpointing'):
        if get(key) is not None and not isinstance(get(key),bool):
            errors.append(f'{key} must be boolean')
    if get('train.amp') and get('train.device')!='cuda':
        errors.append('train.amp requires train.device=cuda')
    checkpoint=get('train.checkpoint_path')
    if checkpoint is not None and not isinstance(checkpoint,str): errors.append('checkpoint_path must be null or a path string')
    return errors
