import numpy as np
import pytest
import torch
from crossvul_dg.losses import CCDROState,GroupDROState,positive_risk_invariance
from crossvul_dg.evaluate import compute_metrics,tune_threshold,recall_at_precision
from crossvul_dg.data import pooled_loader,ClassBalancedBatchSampler
from conftest import loaders


def test_ccdro_initialization_is_order_independent():
    a=CCDROState(eta_q=.1);b=CCDROState(eta_q=.1)
    risks={(0,0):torch.tensor(1.),(0,1):torch.tensor(3.),(1,0):torch.tensor(2.)}
    a.update(risks);b.update(dict(reversed(list(risks.items()))))
    expected=torch.softmax(.1*torch.tensor([1.,3.,2.]),0)
    assert torch.allclose(torch.stack(list(a.q.values())),expected)
    for key in risks:
        assert torch.equal(a.q[key],b.q[key])


def test_extreme_q_is_finite():
    state=GroupDROState(2,eta_q=100.)
    for i in range(4):
        q=state.update(torch.tensor([1e6,0.]))
        assert torch.isfinite(q).all() and torch.allclose(q.sum(),torch.tensor(1.))
    cc=CCDROState(eta_q=100.)
    cc.update({(0,0):torch.tensor(1e6),(0,1):torch.tensor(0.)})
    assert all(torch.isfinite(v) for v in cc.q.values())


def test_positive_variance_and_gradient():
    values=torch.tensor([1.,3.],requires_grad=True)
    v=positive_risk_invariance(list(values))
    assert v.item()==1.
    v.backward()
    assert torch.equal(values.grad,torch.tensor([-1.,1.]))
    assert positive_risk_invariance([torch.tensor(2.)]).item()==0


def test_metrics_threshold_and_confusion():
    y=np.array([0,0,1,1]);p=np.array([.1,.7,.6,.9])
    thr,f1=tune_threshold(y,p)
    m=compute_metrics(y,p,thr,group=np.array(['a','a','b','b']))
    assert m['f1']==pytest.approx(f1)
    assert m['confusion_matrix']==[[1,1],[0,2]]
    assert m['pos_rate']==.5 and m['n']==4 and m['n_pos']==2
    assert 'diagnostic_test_pr_curve_recall_at_p50' in m
    assert set(m['groups'])=={'a','b'}


def test_infeasible_precision_threshold_predicts_nothing():
    y=[1,0];p=[.1,.9]
    threshold,score=tune_threshold(y,p,objective='recall_at_precision',min_precision=1.)
    assert score==0 and threshold>1
    assert compute_metrics(y,p,threshold)['confusion_matrix']==[[1,0],[1,0]]

@pytest.mark.parametrize('y,p',[([],[]),([2],[.5]),([0,1],[.1,float('nan')]),([0,1],[.1]),([0,1],[.1,2])])
def test_metrics_reject_invalid(y,p):
    with pytest.raises(ValueError): compute_metrics(y,p)


def test_single_class_test_is_explicit_but_cannot_select_threshold():
    assert np.isnan(compute_metrics([0,0],[.1,.2])['pr_auc'])
    with pytest.raises(ValueError,match='both classes'): tune_threshold([0,0],[.1,.2])


def test_pooled_erm_is_random_and_visits_each_source_once():
    env,_,_=loaders()
    pooled=pooled_loader(env)
    assert len(pooled.dataset)==24
    assert isinstance(pooled.sampler,torch.utils.data.RandomSampler)
    assert sorted(list(pooled.sampler))==list(range(24))


def test_log_duals_preserve_history_after_probability_underflow():
    state=GroupDROState(2,eta_q=1.)
    state.update(torch.tensor([100000.,0.]))
    state.update(torch.tensor([0.,100000.]))
    assert torch.allclose(state.q,torch.tensor([.5,.5]))
    cc=CCDROState(eta_q=1.)
    cc.update({(0,0):torch.tensor(100000.),(0,1):torch.tensor(0.),(1,1):torch.tensor(1.)})
    # Present-cell probabilities both underflow globally, but conditional objective remains meaningful.
    risk=cc.weighted_loss({(0,1):torch.tensor(2.),(1,1):torch.tensor(4.)})
    assert 2<risk.item()<4
    cc.update({(0,0):torch.tensor(0.),(0,1):torch.tensor(100000.),(1,1):torch.tensor(99999.)})
    assert torch.allclose(torch.stack(list(cc.q.values())),torch.ones(3)/3,atol=1e-6)
