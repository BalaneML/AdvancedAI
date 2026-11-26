"""Explicit future TITAN GPU job; official RAFT426x240 -> RGBA PNG. Never invoked tonight."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import common as c

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',default=str(c.ROOT/'data/titan/flow_tasks.json'));args=parser.parse_args()
 version=c.setup();import torch,numpy as np,cv2
 from PIL import Image
 manifest=json.loads(Path(args.manifest).read_text());tasks=manifest['tasks'];assert manifest['resolution_WH']==[426,240] and manifest['test_read'] is False
 for task in tasks:
  for key in ['first','second','target']:assert Path(task[key]).resolve().is_relative_to(c.ROOT/'data/titan')
 # Fresh process is important: RAFT utils and PTINet utils have the same module name.
 sys.path.insert(0,str(c.ROOT/'RAFT/core'))
 from raft import RAFT
 from utils.utils import InputPadder
 from utils import flow_viz
 net=RAFT(argparse.Namespace(small=False,mixed_precision=False,alternate_corr=False,dropout=0))
 cp=c.ROOT/'RAFT/models/raft-things.pth';cp_hash=c.sha(cp)
 weights=torch.load(cp,map_location='cpu',weights_only=True);net.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()});net.to('cuda').eval();cv2.setNumThreads(1)
 def complete(task):
  p=Path(task['target'])
  if not p.exists():return False
  try:
   with Image.open(p) as image:
    valid=image.mode=='RGBA' and image.size==(426,240);image.verify();return valid
  except Exception:return False
 def read(task):
  arrays=[]
  for key in ['first','second']:
   im=cv2.imread(task[key]);assert im is not None
   arrays.append(cv2.cvtColor(cv2.resize(im,(426,240),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB))
  return arrays
 def save(task,color):
  target=Path(task['target']);target.parent.mkdir(parents=True,exist_ok=True)
  actual=target.with_suffix('.png');tmp=actual.with_name('.'+actual.stem+'.partial.png')
  rgba=np.concatenate([color,np.full((240,426,1),255,dtype=np.uint8)],axis=2);Image.fromarray(rgba).save(tmp,format='PNG',compress_level=3);tmp.replace(actual)
  if target!=actual:
   if target.exists() or target.is_symlink():raise RuntimeError('Preserve existing non-PNG target; inspect before replacing')
   target.symlink_to(actual.name)
 pending=[task for task in tasks if not complete(task)];start=time.monotonic();done=0
 with ThreadPoolExecutor(max_workers=4) as readers,ThreadPoolExecutor(max_workers=4) as writers,torch.no_grad():
  reads=deque();writes=deque();next_index=0
  for _ in range(min(8,len(pending))):reads.append((pending[next_index],readers.submit(read,pending[next_index])));next_index+=1
  while reads:
   task,future=reads.popleft();arrays=future.result()
   if next_index<len(pending):reads.append((pending[next_index],readers.submit(read,pending[next_index])));next_index+=1
   tensors=[torch.from_numpy(a).permute(2,0,1).float()[None].to('cuda') for a in arrays];padder=InputPadder(tensors[0].shape);a,b=padder.pad(*tensors)
   _,flow=net(a,b,iters=20,test_mode=True);flow=padder.unpad(flow)[0].permute(1,2,0).cpu().numpy();assert flow.shape==(240,426,2) and np.isfinite(flow).all()
   writes.append(writers.submit(save,task,flow_viz.flow_to_image(flow)))
   if len(writes)>=8:writes.popleft().result()
   done+=1
   if done%100==0:print('TITAN_FLOW',done,len(pending),'seconds',round(time.monotonic()-start,1),flush=True)
  for future in writes:future.result()
 assert all(complete(task) for task in tasks) and c.sha(cp)==cp_hash
 c.atomic(c.ROOT/'data/titan/flow_complete.json',{'frames':len(tasks),'new_frames':done,'resolution_WH':[426,240],'RGBA':True,'iters':20,'RAFT_checkpoint_sha256':cp_hash,'commit_id':version,'test_read':False,'elapsed_seconds':time.monotonic()-start})
 print('TITAN_FLOW_COMPLETE',flush=True)
if __name__=='__main__':main()
