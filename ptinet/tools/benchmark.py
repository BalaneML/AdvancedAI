"""Batch1, 10 real-window warmup + 100 timed forward calls. Optional RAFT estimate."""
import sys
sys.dont_write_bytecode=True
import argparse,contextlib,json,time,tempfile,subprocess
from pathlib import Path
import runtime as r

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full.yml');p.add_argument('--output',required=True);p.add_argument('--include-raft',action='store_true');p.add_argument('--raft-size',type=int,nargs=2,default=[960,540]);args=p.parse_args();r.clean();cfg=r.config(args.config);r.setup(cfg)
 import torch,numpy as np,csv
 with contextlib.redirect_stdout(sys.stderr):
  torch.manual_seed(0);data=r.dataset(cfg);assert len(data)>=110
  loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,range(110)),batch_size=1,num_workers=cfg['loader_workers'],shuffle=False);net=r.model(cfg);net.eval();times=[]
  with torch.no_grad():
   for index,b in enumerate(loader):
    t=r.inputs(b,cfg);torch.cuda.synchronize();start=time.perf_counter();outputs=r.forward(net,t);torch.cuda.synchronize()
    if index>=10:times.append((time.perf_counter()-start)*1000)
  model_ms=float(np.median(times));raft_ms=0.
  if args.include_raft and cfg['use_opticalflow']:
   pairs=[]
   for paths in data.data.imagefolderpath.iloc[10:110]:
    first=Path(paths[0]);second=first.with_name(f'{int(first.stem)+1:05d}.png')
    if not second.exists():second=first;first=first.with_name(f'{int(first.stem)-1:05d}.png')
    assert first.exists() and second.exists();pairs.append([str(first),str(second)])
   with tempfile.NamedTemporaryFile('w',suffix='.json') as f:
    json.dump(pairs,f);f.flush()
    result=subprocess.check_output([sys.executable,str(r.ROOT/'tools/raft_benchmark.py'),'--pairs',f.name,'--size',*map(str,args.raft_size)],text=True)
   raft_ms=json.loads(result)['raft_median_ms']
 result={'commit_id':r.version(),'batch_size':1,'warmup':10,'samples':100,'model_median_ms':model_ms,'model_p90_ms':float(np.percentile(times,90)),'raft_median_ms':raft_ms if args.include_raft else None,'end_to_end_estimate_ms':model_ms+raft_ms if args.include_raft else None,'Latency_ms':model_ms,'parameters':sum(x.numel() for x in net.parameters()),'PC':torch.cuda.get_device_name(0)}
 path=r.path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=result);w.writeheader();w.writerow(result)
 print(json.dumps(result))
if __name__=='__main__':main()
