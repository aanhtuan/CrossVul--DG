"""Linux resource checks used before costly diagnostics; no background monitor."""
import shutil
from pathlib import Path


def snapshot():
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        memory[key]=int(value.strip().split()[0])*1024
    return dict(available_ram_bytes=memory['MemAvailable'],total_ram_bytes=memory['MemTotal'],
                swap_used_bytes=memory['SwapTotal']-memory['SwapFree'],disk_free_bytes=shutil.disk_usage('.').free)


def require_headroom(min_ram_gib=6,min_disk_gib=5):
    state=snapshot()
    if state['available_ram_bytes']<min_ram_gib*1024**3 or state['disk_free_bytes']<min_disk_gib*1024**3:
        raise RuntimeError(f'Insufficient resources for pretrained-model diagnostic: {state}. '
                           f'Requires {min_ram_gib} GiB available RAM and {min_disk_gib} GiB disk; close memory-heavy apps first.')
    return state


def training_preflight(cfg):
    """Declared conservative budgets; measured watchdog/peak metrics remain required."""
    if cfg['train']['device']!='cuda': return require_headroom()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    state=require_headroom(min_ram_gib=2.5)
    free,total=torch.cuda.mem_get_info()
    length=cfg['encoder']['max_length'];batch=cfg['train']['batch_size']
    # 128M float32 parameters + gradients + two Adam moments; reserve workspace.
    persistent=128_000_000*16
    activation=int(512*1024**2*(length/128)**2*(batch/2))
    if not cfg['train'].get('amp',False): activation*=2
    if not cfg['encoder'].get('gradient_checkpointing',False): activation*=2
    estimate=persistent+activation+512*1024**2
    if free<estimate:
        raise RuntimeError(f'CUDA preflight rejected: free={free}, estimated_required={estimate}; reduce length or use larger GPU')
    return state|dict(cuda_free_bytes=free,cuda_total_bytes=total,estimated_cuda_required_bytes=estimate,
                      estimate_note='Conservative estimate; a guarded measured step is required')
