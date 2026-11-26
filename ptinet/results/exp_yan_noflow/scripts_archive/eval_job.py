"""Evaluate a study checkpoint with all val series."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import common as c

def paired_compare(a,b,out):
 import numpy as np,pandas as pd
 from scipy.stats import wilcoxon
 assert np.array_equal(a[['video_id','start_frame','dataset_id','label']].to_numpy(),b[['video_id','start_frame','dataset_id','label']].to_numpy())
 delta=b.ADE.to_numpy()-a.ADE.to_numpy();rng=np.random.default_rng(0)
 p=float(wilcoxon(delta,alternative='two-sided',zero_method='wilcox').pvalue) if np.any(delta!=0) else 1.
 boot=np.mean(rng.choice(delta,size=(10000,len(delta)),replace=True),axis=1)
 grouped=pd.DataFrame({'video':a.video_id,'delta':delta}).groupby('video').delta.agg(['sum','count'])
 sample=rng.integers(0,len(grouped),size=(10000,len(grouped)))
 cluster=grouped['sum'].to_numpy()[sample].sum(axis=1)/grouped['count'].to_numpy()[sample].sum(axis=1)
 row={'N':len(delta),'mean_ADE_difference_noflow_minus_D1':float(delta.mean()),'wilcoxon_p_two_sided':p,'paired_bootstrap_CI95_low':float(np.quantile(boot,.025)),'paired_bootstrap_CI95_high':float(np.quantile(boot,.975)),'video_cluster_bootstrap_CI95_low':float(np.quantile(cluster,.025)),'video_cluster_bootstrap_CI95_high':float(np.quantile(cluster,.975)),'bootstrap_iterations':10000,'seed':0,'commit_id':c.commit()}
 pd.DataFrame([row]).to_csv(out/'paired_test.csv',index=False,float_format='%.17g')
 pd.DataFrame({'series_index':a.series_index,'video_id':a.video_id,'D1_ADE':a.ADE,'noflow_ADE':b.ADE,'difference':delta}).to_csv(out/'paired_samples.csv',index=False,float_format='%.17g')
 return row

def main():
 parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args();start=time.monotonic();version=c.setup()
 import pandas as pd
 config=json.loads(Path(args.config).read_text());run=Path(config['out_dir'])/config['log_name'];out=Path(config['study_report'])
 cp=run/f"model_best{config['lr']}_{config['hidden_size']}.pkl"
 frame,row=c.evaluate(config,cp,out/config['log_name'])
 row.update(run=config['log_name'],training_commit_id=json.loads((run/'launch.json').read_text())['commit_id'],epochs=config['n_epochs'],train_video_fraction=config['train_video_fraction'],image_resize_width=config['image_resize_width'],image_resize_height=config['image_resize_height'])
 pd.DataFrame([row]).to_csv(out/(config['log_name']+'_summary.csv'),index=False,float_format='%.17g')
 if config.get('study_role')=='jaad_full_noflow':
  base=pd.read_csv(c.OUT/'step3_masking/base/samples.csv',float_precision='round_trip')
  prior=pd.read_csv(c.OUT/'step3_masking/base/summary.csv',float_precision='round_trip').iloc[0].to_dict();prior.update(run='D1 Baseline',training_commit_id='pre-git; snapshot '+(c.OUT/'step1_prepare/baseline_commit.txt').read_text().strip())
  pd.DataFrame([prior,row]).to_csv(out/'summary.csv',index=False,float_format='%.17g')
  stats=paired_compare(base,frame,out)
  groupframes=[]
  for condition,f in [('D1',base),('noflow',frame)]:
   for g in c.grouped(f):groupframes.append({'condition':condition,**g,'commit_id':version})
  pd.DataFrame(groupframes).to_csv(out/'groups.csv',index=False,float_format='%.17g')
  note='\n\n対応検定結果: '+json.dumps(stats,ensure_ascii=False)+'。Wilcoxonは系列単位の両側検定。重複窓・同動画の相関があるため独立系列bootstrapに加えて動画単位cluster bootstrap CIを併記する。p>0.05だけでは同等性を証明しない。非劣性許容幅は指定されていないため、厳密な非劣性成立は主張しない。'
 else:note=''
 c.readme(out,config.get('study_title',config['log_name']),f'訓練コマンド: `{c.PYTHON} {c.ROOT}/scripts/flow_study/train_job.py {args.config}`。評価: `{c.PYTHON} {Path(__file__).resolve()} {args.config}`。評価時間{time.monotonic()-start:.1f}秒。設定・所要時間はrun launch.json/complete.json、各epochはepochs.json。\n\n{json.dumps(row,ensure_ascii=False)}\n\nseedはモデル生成前reset、batch2、workers8、全val、eval/no_grad、平均ADE/FDE、標準F1/Accuracyとseries max P(crossing)のAPを報告。既存の作者F1式は使わない。'+note)
 c.atomic(out/(config['log_name']+'_evaluation_complete.json'),{'status':'complete','commit_id':version,'elapsed_seconds':time.monotonic()-start})
if __name__=='__main__':main()
