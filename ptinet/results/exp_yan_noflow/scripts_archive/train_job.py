"""Run a committed study config through the author's training function/main; no test."""
import sys
sys.dont_write_bytecode=True
import argparse,builtins,json,time,os,runpy,traceback
from pathlib import Path
import common as c

def main():
 parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
 started=time.monotonic();version=c.setup();import torch,pandas as pd,numpy as np
 config=json.loads(Path(args.config).read_text());run=Path(config['out_dir'])/config['log_name'];assert not run.exists(),'Existing run, refuse overwrite'
 run.mkdir(parents=True);report=Path(config['study_report']);report.mkdir(parents=True,exist_ok=True)
 c.atomic(run/'launch.json',{'commit_id':version,'config':config,'started_at':c.now(),'script_sha256':c.sha(__file__)})
 os.environ['PTINET_CONFIG']=str(Path(args.config).resolve());os.environ['TORCH_HOME']=str(c.ROOT/'downloads/torch_hub');os.environ.pop('PTINET_EVAL_ONLY',None)
 # train.py's config loader accepts JSON (YAML subset).
 os.chdir(run);sys.argv=[str(c.PROJECT/'train.py')]
 original_print=builtins.print;last=None;epochs=[]
 def observed(*items,**kw):
  nonlocal last
  original_print(*items,**kw)
  if items and items[0]=='Training ...':last=time.monotonic()
  if items and items[0]=='e:':
   t=time.monotonic();entry={'epoch':int(items[1]),'epoch_number':int(items[1])+1,'seconds':t-last,'at':c.now()};last=t;epochs.append(entry)
   reference=float(np.median([row['seconds'] for row in epochs[-5:]]))
   estimate=time.time()+(config['n_epochs']-len(epochs))*reference
   c.atomic(run/'epochs.json',epochs);c.atomic(report/'training_status.json',{'phase':'training','run':config['log_name'],'epochs_completed':len(epochs),'epochs_total':config['n_epochs'],'commit_id':version,'latest_epoch':entry,'recent_epoch_seconds_median':reference,'estimated_training_finish_unix':estimate})
   root_status=c.OUT/'status.json'
   if root_status.exists():
    global_status=json.loads(root_status.read_text());global_status.update(updated_at=c.now(),active_training={'run':config['log_name'],'epochs_completed':len(epochs),'epochs_total':config['n_epochs'],'recent_epoch_seconds_median':reference,'estimated_training_finish_unix':estimate});c.atomic(root_status,global_status)
   handover=c.ROOT/'HANDOVER.md';text=handover.read_text();marker='- 训练进度（自动更新）：'
   line=f"{marker}{config['log_name']} {len(epochs)}/{config['n_epochs']} epoch，更新{c.now()}，预计训练结束UNIX={estimate:.0f}。"
   if marker in text:text='\n'.join(line if x.startswith(marker) else x for x in text.split('\n'))
   elif '<!-- FLOW_STUDY_END -->' in text:text=text.replace('<!-- FLOW_STUDY_END -->',line+'\n<!-- FLOW_STUDY_END -->')
   handover.write_text(text)
   original_print('FLOW_EPOCH',json.dumps(entry),flush=True)
 builtins.print=observed
 try:
  namespace=runpy.run_path(str(c.PROJECT/'train.py'),run_name='__main__');assert namespace['args'].skip_test
 finally:builtins.print=original_print
 assert c.commit()==version
 history=pd.read_csv(run/f"data_final{config['lr']}_{config['hidden_size']}.csv")
 assert len(history)==config['n_epochs'] and np.isfinite(history.select_dtypes('number').to_numpy()).all()
 c.atomic(run/'complete.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,'commit_id':version,'epochs':len(history),'config':config,'test_read':False})
 print('TRAIN_COMPLETE',config['log_name'],flush=True)
if __name__=='__main__':main()
