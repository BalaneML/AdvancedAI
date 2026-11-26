"""Prepare exact TITAN flow needs from selected train windows and ALL val windows; CPU only."""
import os
for key in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:os.environ[key]=''
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import common as c

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--processed',default=str(c.ROOT/'data/titan/processed'));parser.add_argument('--manifest',default=str(c.ROOT/'data/titan/flow_tasks.json'));args=parser.parse_args()
 version=c.setup(gpu=False);source=Path(args.processed).resolve();assert source.is_relative_to(c.ROOT/'data/titan')
 assert (source/'preparation_audit.json').exists(),'Run local train/val preprocessing first'
 cfg=c.cfg(dataset='titan',data_dir=str(source),input=5,output=5,stride=5,use_image=False,use_opticalflow=False,train_video_fraction=.25)
 paths=set();counts={};sources={}
 for split in ['train','val']:
  data=c.data(cfg,split);counts[split]=len(data);sources[split]=[Path(p).stem for p in data.source_files]
  for sequence in data.data.img_file:paths.update(sequence)
 tasks=[];folders={}
 for name in sorted(paths):
  image=Path(name);assert image.is_file() and image.resolve().is_relative_to(c.ROOT/'data/titan')
  if image.parent not in folders:
   files=sorted([p for p in image.parent.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg']],key=lambda p:(float(p.stem),p.name));folders[image.parent]=(files,{p:i for i,p in enumerate(files)})
  files,indices=folders[image.parent];i=indices[image];assert len(files)>1
  a,b=(image,files[i+1]) if i+1<len(files) else (files[i-1],image)
  target=Path(str(image).replace('/images/','/opticalflow/'));assert target!=image
  tasks.append({'first':str(a),'second':str(b),'target':str(target)})
 c.atomic(args.manifest,{'tasks':tasks,'sample_counts':counts,'source_videos':sources,'resolution_WH':[426,240],'iters':20,'commit_id':version,'test_read':False})
 print('TITAN_FLOW_MANIFEST_READY',len(tasks),'frames; train/val series',counts,flush=True)
if __name__=='__main__':main()
