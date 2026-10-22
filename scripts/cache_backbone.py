"""Cache one immutable CodeBERT revision, one download worker, no training."""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from pathlib import Path
import json
from huggingface_hub import snapshot_download
import yaml
config=yaml.safe_load(Path('configs/default.yaml').read_text())
path=snapshot_download(config['encoder']['name'],revision=config['encoder']['revision'],max_workers=1,
                       allow_patterns=['config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','special_tokens_map.json','pytorch_model.bin','model.safetensors'])
print(json.dumps(dict(snapshot=path,model=config['encoder']['name'],revision=config['encoder']['revision'])))
