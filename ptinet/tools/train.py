"""Run the corrected author training loop, with relocatable config and test disabled."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,runpy,tempfile
from pathlib import Path
import runtime as r

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full.yml');p.add_argument('--epochs',type=int);args=p.parse_args();r.clean();cfg=r.config(args.config);r.setup(cfg)
 if args.epochs is not None:cfg['n_epochs']=args.epochs
 target=Path(cfg['out_dir'])/cfg['log_name'];assert not target.exists(),'Preserve existing run; choose another log_name'
 target.mkdir(parents=True);(target/'launch.json').write_text(json.dumps({'config':cfg,'commit_id':r.version()},indent=2))
 with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:json.dump(cfg,f);name=f.name
 os.environ['PTINET_CONFIG']=name;os.environ.pop('PTINET_EVAL_ONLY',None);os.chdir(target);sys.argv=['train.py']
 try:runpy.run_path(str(r.ROOT/'train.py'),run_name='__main__')
 finally:Path(name).unlink(missing_ok=True)
if __name__=='__main__':main()
