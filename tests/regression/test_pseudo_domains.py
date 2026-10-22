import pytest
import torch
from crossvul_dg.pseudo_domains import cluster_tail_projects,project_mean_embedding
from conftest import TinyModel,Tokenizer,frame
import pandas as pd


def test_pseudo_embeddings_move_model_before_inputs():
    class Model(TinyModel):
        moved=False
        def to(self,device):
            self.moved=True
            return super().to(device)
        def encode(self,ids,mask):
            assert self.moved
            assert ids.device==next(self.parameters()).device
            return super().encode(ids,mask)
    model=Model()
    tail=pd.concat([frame('tail_a'),frame('tail_b')],ignore_index=True)
    mapping=cluster_tail_projects(model,tail,Tokenizer(),n_clusters=1,max_samples_per_proj=3,max_length=12)
    assert set(mapping)=={'tail_a','tail_b'}
    with pytest.raises(ValueError): cluster_tail_projects(model,tail,Tokenizer(),n_clusters=3)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA PyTorch/device unavailable in CPU verification environment')
def test_native_cuda_pseudo_embeddings():
    embedding=project_mean_embedding(TinyModel(),frame('tail'),Tokenizer(),device='cuda',max_length=12,batch_size=2)
    assert embedding.shape==(8,)
