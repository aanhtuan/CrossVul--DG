import copy
import pytest
import torch
from torch import nn
from crossvul_dg.trainers import TrainerConfig,build_trainer,capture_rng,restore_rng
from crossvul_dg.losses import CCDROState,bce_per_sample,class_conditional_cell_losses,positive_risk_invariance
from crossvul_dg.utils import set_seed
from conftest import TinyModel,loaders


def test_amp_refuses_cpu_instead_of_silently_changing_precision():
    with pytest.raises(ValueError,match='requires CUDA'):
        build_trainer('erm',TinyModel(),TrainerConfig(amp=True,device='cpu'))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native CUDA runtime required')
@pytest.mark.parametrize('checkpointed_transformer',[False,True])
def test_native_fp16_replay_matches_full_graph(checkpointed_transformer):
    set_seed(123);env,_,_=loaders(balanced=True);batches=[next(iter(l)) for l in env]
    if checkpointed_transformer:
        from transformers import RobertaConfig,RobertaModel
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                cfg=RobertaConfig(vocab_size=66,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
                                  intermediate_size=32,hidden_dropout_prob=.2,attention_probs_dropout_prob=.2)
                cfg._attn_implementation='eager'
                self.encoder=RobertaModel(cfg)
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                self.classifier=nn.Linear(16,1)
            def forward(self,ids,mask):return self.classifier(self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]).squeeze(-1)
        base=Model().cuda()
    else:base=TinyModel().cuda()
    trainer=build_trainer('cc-dro',copy.deepcopy(base),TrainerConfig(method='cc-dro',amp=True,device='cuda'))
    trainer.dro=CCDROState(device='cuda');state=capture_rng();cells={}
    for i,batch in enumerate(batches):
        with torch.autocast('cuda',dtype=torch.float16):logits=base(batch['input_ids'].cuda(),batch['attention_mask'].cuda()).float()
        cells.update(class_conditional_cell_losses(bce_per_sample(logits,batch['label'].cuda()),batch['label'].cuda(),i))
    dual=CCDROState(device='cuda');dual.update(cells)
    loss=dual.weighted_loss(cells)+.1*positive_risk_invariance([v for (p,c),v in cells.items() if c==1])
    (loss*trainer.scaler.get_scale()).backward()
    for parameter in base.parameters():
        if parameter.grad is not None:parameter.grad.div_(trainer.scaler.get_scale())
    restore_rng(state);trainer._coupled_step(batches,.1);trainer.scaler.unscale_(trainer.optim)
    for expected,actual in zip(base.parameters(),trainer.model.parameters()):
        if expected.grad is None:assert actual.grad is None
        else:assert torch.allclose(expected.grad,actual.grad,atol=2e-4,rtol=5e-3)
