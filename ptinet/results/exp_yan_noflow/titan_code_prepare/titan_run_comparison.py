"""Explicit future TITAN comparison entry point. Not part of tonight's queue."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time
from pathlib import Path
import common as c

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--include-seed2',action='store_true');args=parser.parse_args()
 processed=c.ROOT/'data/titan/processed';flow=c.ROOT/'data/titan/flow_complete.json'
 assert (processed/'preparation_audit.json').exists() and flow.exists(),'Real local data preparation and RAFT must finish first'
 assert json.loads(flow.read_text())['test_read'] is False
 version=c.setup();out=c.OUT/'step8_titan';out.mkdir(exist_ok=True)
 import pandas as pd,numpy as np
 from eval_job import paired_compare
 seeds=[0,1,2] if args.include_seed2 else [0,1];records=[];sample_frames={}
 for seed in seeds:
  for tag in ['baseline','noflow']:
   template=c.OUT/'titan_code_prepare/templates'/f'titan_small_{tag}_out5_s{seed}.json'
   cfg=json.loads(template.read_text());assert cfg['dataset']=='titan' and cfg['use_attribute'] and cfg['input']==5
   run=Path(cfg['out_dir'])/cfg['log_name'];assert not run.exists(),'Existing TITAN run found; preserve and inspect instead of overwriting'
   for script,log,timeout in [('train_job.py',out/(cfg['log_name']+'_train.log'),36000),('eval_job.py',out/(cfg['log_name']+'_eval.log'),1800)]:
    with log.open('w') as stream:subprocess.run([sys.executable,str(Path(__file__).parent/script),str(template)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
   row=pd.read_csv(out/(cfg['log_name']+'_summary.csv'),float_precision='round_trip').iloc[0].to_dict();row.update(variant=tag,seed=seed);records.append(row)
   sample_frames[(tag,seed)]=pd.read_csv(out/cfg['log_name']/'samples.csv',float_precision='round_trip')
  destination=out/f'paired_seed{seed}';destination.mkdir(exist_ok=True)
  paired_compare(sample_frames[('baseline',seed)],sample_frames[('noflow',seed)],destination)
 per_seed=pd.DataFrame(records);per_seed.to_csv(out/'per_seed.csv',index=False,float_format='%.17g')
 summaries=[]
 for tag in ['baseline','noflow']:
  part=per_seed[per_seed.variant==tag];row={'variant':tag,'seed_count':len(part),'commit_id':version}
  for metric in ['ADE','FDE','F1','Accuracy','PR_AUC']:row[metric]=float(part[metric].mean());row[metric+'_sd']=float(part[metric].std(ddof=1))
  summaries.append(row)
 # Average ADE/FDE over seeds per fixed validation series for an additional paired comparison.
 means={}
 for tag in ['baseline','noflow']:
  reference=sample_frames[(tag,seeds[0])].copy()
  for seed in seeds[1:]:assert np.array_equal(reference[['video_id','start_frame','dataset_id','label']],sample_frames[(tag,seed)][['video_id','start_frame','dataset_id','label']])
  for metric in ['ADE','FDE']:reference[metric]=np.mean([sample_frames[(tag,seed)][metric].to_numpy() for seed in seeds],axis=0)
  means[tag]=reference
 destination=out/'paired_seed_mean';destination.mkdir(exist_ok=True);paired_compare(means['baseline'],means['noflow'],destination)
 # Time the small TITAN model itself (5 observation frames), not JAAD's 16-frame benchmark.
 import torch,gc
 for row,tag in zip(summaries,['baseline','noflow']):
  cfg=json.loads((c.OUT/'titan_code_prepare/templates'/f'titan_small_{tag}_out5_s0.json').read_text());torch.manual_seed(0)
  dataset=c.data(cfg);assert len(dataset)>10
  n=min(100,len(dataset)-10);loader=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,range(n+10)),batch_size=1,num_workers=8,shuffle=False)
  cp=Path(cfg['out_dir'])/cfg['log_name']/'model_best0.001_512.pkl';net=c.model(cfg,cp);net.eval();times=[]
  with torch.no_grad():
   for index,b in enumerate(loader):
    t=c.inputs(b);torch.cuda.synchronize();start=time.perf_counter();outputs=c.forward(net,t);torch.cuda.synchronize()
    if index>=10:times.append((time.perf_counter()-start)*1000)
  row.update(model_median_ms=float(np.median(times)),model_p90_ms=float(np.percentile(times,90)),timing_samples=n)
  del net,dataset,outputs,t;gc.collect();torch.cuda.empty_cache()
 pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False,float_format='%.17g')
 c.readme(out,'TITAN縮小設定・複数seed',f'実行コマンド: `{c.PYTHON} {Path(__file__).resolve()}`。seed={seeds}、22epoch、25%動画、画像小サイズはtemplatesに明記。各指標はseed平均±標本SD(ddof=1)。per_seed.csvと各runの全val結果に基づく。モデル時間はTITAN5観測frame・batch1で10warm-up後最大100系列、RAFTを含まない。\n\npaired_seed*/paired_test.csvに各seedのWilcoxonとpaired/video-cluster bootstrap 95%CI、paired_seed_meanには固定val系列ごとのseed平均ADEで対応検定を記録。旧helper列名のD1はここではTITAN Baselineを指す（JAAD D1と比較しない）。\n\nTITANの公開ラベルはwalking/runningの代理と既存系列除外を保持しており、JAADの実横断ラベルと意味が同一とは主張しない。test未使用、データは本機のみ。')
 c.atomic(out/'status.json',{'status':'complete','seeds':seeds,'commit_id':version,'test_read':False})
 print('TITAN_COMPARISON_COMPLETE',flush=True)
if __name__=='__main__':main()
