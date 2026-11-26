import sys
sys.dont_write_bytecode=True
import argparse,json,os,time,runpy
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,pandas as pd,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/JAAD'
from functools import lru_cache
@lru_cache(maxsize=48)
def load_frame(path):
 im=cv2.imread(str(path));assert im is not None
 return cv2.cvtColor(cv2.resize(im,(960,540),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
def load_pair(task):return load_frame(task[0]),load_frame(task[1])
def load_model():
    sys.path.insert(0,str(ROOT/'third_party/RAFT/core'))
    from raft import RAFT
    from utils.utils import InputPadder
    from utils import flow_viz
    model=RAFT(argparse.Namespace(small=False,mixed_precision=False,alternate_corr=False,dropout=0))
    weights=torch.load(ROOT/'third_party/RAFT/models/raft-things.pth',map_location='cpu',weights_only=True)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True)
    model.to('cuda').eval()
    def infer(pair):
        arrays=[torch.from_numpy(a).permute(2,0,1).float()[None].to('cuda') for a in pair]
        padder=InputPadder(arrays[0].shape)
        first,second=padder.pad(*arrays)
        with torch.no_grad(): _,flow=model(first,second,iters=20,test_mode=True)
        flow=padder.unpad(flow)[0].permute(1,2,0).cpu().numpy()
        assert flow.shape==(540,960,2) and np.isfinite(flow).all()
        return flow
    return infer,flow_viz

def calculate(tasks,infer,flow_viz,label):
    def save(flow,target):
        rgb=flow_viz.flow_to_image(flow)
        rgba=np.concatenate((rgb,np.full((540,960,1),255,dtype=np.uint8)),axis=2)
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_name('.'+target.stem+'.partial.png')
        Image.fromarray(rgba).save(tmp,format='PNG',compress_level=3)
        os.replace(tmp,target)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as readers,ThreadPoolExecutor(max_workers=4) as writers:
        next_index=0
        reads,writes=deque(),deque()
        for _ in range(min(8,len(tasks))):
            reads.append((tasks[next_index],readers.submit(load_pair,tasks[next_index])))
            next_index+=1
        processed=0
        while reads:
            task,future=reads.popleft()
            pair=future.result()
            if next_index<len(tasks):
                reads.append((tasks[next_index],readers.submit(load_pair,tasks[next_index])))
                next_index+=1
            flow=infer(pair)
            writes.append(writers.submit(save,flow,task[2]))
            if len(writes)>=8: writes.popleft().result()
            processed+=1
            if processed%50==0:
                print(f'FLOW {label} progress {processed}/{len(tasks)} elapsed={time.monotonic()-started:.1f}s',flush=True)
        for future in writes: future.result()
    return time.monotonic()-started

def main():
 import xml.etree.ElementTree as ET
 assert torch.cuda.is_available(),'GPU required for official RAFT';torch.set_num_threads(4);cv2.setNumThreads(1)
 requested=set()
 for split in ['train','val']:
  for path in sorted((DATA/'PN_ego'/split).glob('*.csv')):requested.update(pd.read_csv(path,usecols=['imagefolderpath']).imagefolderpath)
 assert requested,'Prepare annotation CSVs first'
 tasks=[];counts={}
 for source in sorted(requested):
  source=Path(source);video=source.parent.name
  if video not in counts:counts[video]=int(ET.parse(DATA/'annotations'/(video+'.xml')).find('./meta/task/size').text)
  i=int(source.stem);a,b=(source,source.with_name(f'{i+1:05d}.png')) if i+1<counts[video] else (source.with_name(f'{i-1:05d}.png'),source)
  target=DATA/'opticalflow'/video/source.name
  if target.exists():
   with Image.open(target) as im:assert im.mode=='RGBA' and im.size==(960,540);im.verify()
   continue
  assert a.exists() and b.exists();tasks.append((a,b,target))
 if tasks:
  infer,flow_viz=load_model();seconds=calculate(tasks,infer,flow_viz,'train_val');print('RAFT_COMPLETE',len(tasks),seconds)
if __name__=='__main__':main()
