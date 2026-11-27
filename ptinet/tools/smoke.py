"""A tiny real-data training/val/checkpoint/evaluation check; never test."""
import sys
sys.dont_write_bytecode=True
import argparse,os,json,time
from pathlib import Path
from types import SimpleNamespace
import runtime as r

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/small.yml');p.add_argument('--device');args=p.parse_args();r.clean();cfg=r.config(args.config,args.device);r.setup(cfg)
 import torch,train
 torch.manual_seed(0);run=Path(cfg['out_dir'])/('smoke_'+str(time.time_ns()));cfg.update(n_epochs=1,out_dir=str(run.parent),log_name=run.name)
 tr=torch.utils.data.Subset(r.dataset(cfg,'train'),range(4));va=torch.utils.data.Subset(r.dataset(cfg,'val'),range(4));run.mkdir(parents=True)
 old=Path.cwd();os.chdir(run)
 try:net=train.train(SimpleNamespace(**cfg),tr,va)
 finally:os.chdir(old)
 cp=run/'model_best0.001_512.pkl';assert cp.exists() and net.training
 result,_=r.evaluate(cfg,cp,4);(run/'smoke_result.json').write_text(json.dumps(result,indent=2));print('SMOKE_PASSED',run)
if __name__=='__main__':main()
