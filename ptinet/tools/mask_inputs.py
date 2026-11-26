"""Full-forward paired input masks; no model-branch feature caching."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,contextlib
from pathlib import Path
import runtime as r

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full.yml');p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);args=p.parse_args();r.clean();cfg=r.config(args.config);root=r.path(args.output_dir);assert not root.exists(),'Preserve existing results';root.mkdir(parents=True)
 import torch
 original=r.forward;results=[]
 for condition in ['base','zero_flow','zero_image']:
  def forward(net,t):
   if condition=='zero_flow':t={**t,'optical':torch.zeros_like(t['optical'])}
   if condition=='zero_image':t={**t,'image':torch.zeros_like(t['image'])}
   return original(net,t)
  r.forward=forward
  # evaluate resets seed before data/model construction for every condition.
  with contextlib.redirect_stdout(sys.stderr):summary,samples=r.evaluate(cfg,r.path(args.checkpoint))
  summary['condition']=condition;results.append(summary)
  with (root/(condition+'_samples.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=samples[0]);w.writeheader();w.writerows(samples)
 r.forward=original
 with (root/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
 print('Masks complete',root)
if __name__=='__main__':main()
