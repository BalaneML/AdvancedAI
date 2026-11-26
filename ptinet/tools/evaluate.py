"""One-row CSV, fixed columns, existing checkpoint, validation only."""
import sys
sys.dont_write_bytecode=True
import argparse,contextlib,csv
from pathlib import Path
import runtime as r

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full.yml');p.add_argument('--checkpoint',required=True);p.add_argument('--output');p.add_argument('--samples-output');p.add_argument('--reference');p.add_argument('--tolerance',type=float,default=.05);p.add_argument('--seed',type=int);p.add_argument('--device');p.add_argument('--limit',type=int);args=p.parse_args()
 r.clean();cfg=r.config(args.config,args.device,args.seed)
 with contextlib.redirect_stdout(sys.stderr):result,samples=r.evaluate(cfg,r.path(args.checkpoint),args.limit)
 if args.reference:
  with r.path(args.reference).open() as f:reference=list(csv.DictReader(f))[0]
  for key in ['ADE','FDE']:assert abs(result[key]-float(reference[key]))<=args.tolerance,f'{key} mismatch; inspect data/environment before experiments'
  assert abs(result['F1']-float(reference['F1']))<1e-8 and abs(result['Accuracy']-float(reference['Accuracy']))<1e-8
 def output(f):
  w=csv.DictWriter(f,fieldnames=r.FIELDS);w.writeheader();w.writerow(result)
 if args.output:
  path=r.path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('w') as f:output(f)
 else:output(sys.stdout)
 if args.samples_output:
  path=r.path(args.samples_output);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
if __name__=='__main__':main()
