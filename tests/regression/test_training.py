import copy
import pytest
import torch
from crossvul_dg.trainers import (TrainerConfig,build_trainer,capture_rng,restore_rng)
from crossvul_dg.losses import (bce_per_sample,bce_mean,class_conditional_cell_losses,
                              positive_risk_invariance,CCDROState,GroupDROState)
from crossvul_dg.utils import set_seed
from conftest import TinyModel,loaders

@pytest.mark.parametrize('method',['cc-dro','dro'])
def test_replay_gradient_matches_full_graph_with_dropout(method):
    set_seed(12)
    env,_,_=loaders(balanced=True)
    batches=[next(iter(l)) for l in env]
    base=TinyModel();base.train()
    trainer=build_trainer(method,copy.deepcopy(base),TrainerConfig(method=method,cc_eta_q=.03,dro_eta_q=.03))
    rng=capture_rng()
    cells={}
    for i,batch in enumerate(batches):
        logits=base(batch['input_ids'],batch['attention_mask'])
        values=class_conditional_cell_losses(bce_per_sample(logits,batch['label']),batch['label'],i) if method=='cc-dro' else {(i,0):bce_mean(logits,batch['label'])}
        cells.update(values)
    if method=='cc-dro':
        state=CCDROState(.03);state.update(cells)
        loss=state.weighted_loss(cells)+.7*positive_risk_invariance([v for (p,c),v in cells.items() if c==1])
        trainer.dro=CCDROState(.03)
    else:
        state=GroupDROState(2,.03);risk=torch.stack(list(cells.values()));state.update(risk)
        loss=state.weighted_loss(risk);trainer.dro=GroupDROState(2,.03)
    loss.backward()
    expected_rng=capture_rng()
    restore_rng(rng)
    log=trainer._coupled_step(batches,.7 if method=='cc-dro' else 0.)
    assert log['loss']==pytest.approx(loss.item(),abs=1e-6)
    for a,b in zip(base.parameters(),trainer.model.parameters()):
        assert torch.allclose(a.grad,b.grad,atol=2e-7,rtol=2e-5)
    assert torch.equal(capture_rng()['torch'],expected_rng['torch'])

@pytest.mark.parametrize('method',['erm','irm','dro','cc-dro','pd-dro','da-coral','da-dann'])
def test_all_methods_execute_and_choose_validation_model(method):
    set_seed(3);env,val,target=loaders(balanced=method=='cc-dro')
    cfg=TrainerConfig(method=method,epochs=1,lr=.01,irm_warmup_epochs=0,cc_warmup_epochs=0)
    t=build_trainer(method,TinyModel(),cfg,target_loader=target)
    t.fit(env,val)
    assert t.best_state and len(t.history)==1
    assert torch.isfinite(torch.tensor(t.evaluate(val)['pr_auc']))
    assert len(t._loaders)==1 if method in ('erm','da-coral','da-dann') else len(t._loaders)==2
    if method=='cc-dro':
        assert all(k in t.history[0] for k in ('classification_loss','invariance_penalty','q_entropy','gradient_norm'))

class Interrupted(Exception): pass

@pytest.mark.parametrize('method',['erm','cc-dro','da-dann'])
def test_epoch_resume_is_exact_including_algorithm_and_rng(tmp_path,method):
    def new(checkpoint=None):
        set_seed(19);env,val,target=loaders(balanced=method=='cc-dro')
        cfg=TrainerConfig(method=method,epochs=3,lr=.01,cc_warmup_epochs=0,early_stopping_patience=9,
                          checkpoint_path=str(checkpoint) if checkpoint else None)
        return build_trainer(method,TinyModel(),cfg,target_loader=target),env,val
    full,env,val=new();full.fit(env,val)
    path=tmp_path/'latest.pt'
    interrupted,env,val=new(path)
    original=interrupted._save_checkpoint
    def stop(epoch):
        original(epoch)
        raise Interrupted()
    interrupted._save_checkpoint=stop
    with pytest.raises(Interrupted): interrupted.fit(env,val)
    assert path.exists() and (tmp_path/'latest.best.pt').exists()
    resumed,env,val=new(path);resumed.fit(env,val)
    assert full.history==resumed.history
    for key,value in full.model.state_dict().items():
        assert torch.equal(value,resumed.model.state_dict()[key])
    assert full.scheduler.state_dict()==resumed.scheduler.state_dict()


def test_stale_checkpoint_config_dataset_and_model_are_rejected(tmp_path):
    set_seed(9);env,val,_=loaders()
    cfg=TrainerConfig(epochs=1,checkpoint_path=str(tmp_path/'latest.pt'))
    build_trainer('erm',TinyModel(),cfg).fit(env,val)
    for change in ('lr','data','model'):
        env,val,_=loaders();other=copy.deepcopy(cfg);model=TinyModel()
        if change=='lr': other.lr=.02
        if change=='data': env[0].dataset.df.loc[0,'func']='changed source'
        if change=='model': model.model_name='other-backbone'
        with pytest.raises(ValueError,match='identity mismatch'):
            build_trainer('erm',model,other).fit(env,val)


def test_latest_resume_and_best_selection_are_distinct(tmp_path):
    set_seed(23);env,val,_=loaders()
    path=tmp_path/'latest.pt'
    cfg=TrainerConfig(epochs=3,early_stopping_patience=1,checkpoint_path=str(path))
    trainer=build_trainer('erm',TinyModel(),cfg)
    trainer._val_monitor=lambda _: .5
    trainer.fit(env,val)
    latest=torch.load(path,weights_only=False)
    best=torch.load(tmp_path/'latest.best.pt',weights_only=False)
    assert latest['completed_epoch']==1 and latest['patience_left']==0
    assert all(torch.equal(v,best['model_state_dict'][k]) for k,v in trainer.model.state_dict().items())
    resumed=build_trainer('erm',TinyModel(),cfg)
    resumed._val_monitor=lambda _:pytest.fail('early-stopped run must not train again')
    resumed.fit(env,val)
    assert len(resumed.history)==2
