"""Run one child job with a RAM/swap/runtime watchdog and peak resource evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossvul_dg.resources import snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--timeout_seconds',type=int,default=1200)
    parser.add_argument('--min_available_gib',type=float,default=1.)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command: parser.error('Child command required after --')
    args.out.mkdir(parents=True,exist_ok=True);start=time.monotonic();initial=snapshot()
    peak_rss=peak_gpu=0;minimum=initial['available_ram_bytes'];reason=None;low=0
    with (args.out/'job.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                state=snapshot();minimum=min(minimum,state['available_ram_bytes'])
                try:
                    for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                        if line.startswith(('VmHWM:','VmRSS:')):peak_rss=max(peak_rss,int(line.split()[1])*1024)
                except FileNotFoundError:pass
                try:
                    usage=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                    for line in usage.splitlines():
                        pid,mem=line.split(',')
                        if int(pid)==process.pid:peak_gpu=max(peak_gpu,int(mem.strip())*1024**2)
                except (OSError,ValueError,subprocess.SubprocessError):pass
                low=low+1 if state['available_ram_bytes']<args.min_available_gib*1024**3 else 0
                if low>=3:reason='available RAM below safety floor for three samples'
                if state['swap_used_bytes']-initial['swap_used_bytes']>1024**3:reason='system swap grew more than 1 GiB'
                if time.monotonic()-start>args.timeout_seconds:reason='runtime budget exceeded'
                if reason:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                    break
                time.sleep(2)
        finally:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
            exit_code=process.wait()
    report=dict(command=command,exit_code=exit_code,abort_reason=reason,runtime_seconds=time.monotonic()-start,
                peak_child_rss_bytes=peak_rss,peak_child_gpu_bytes=peak_gpu,min_system_available_ram_bytes=minimum,
                initial_resources=initial,final_resources=snapshot(),sampling_seconds=2)
    (args.out/'resources.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if exit_code==0 and reason is None else 1)

if __name__=='__main__':main()
