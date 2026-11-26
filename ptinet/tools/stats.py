"""Paired validation ADE/FDE summaries with IID and video-cluster bootstrap."""
import argparse,csv
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--experiment',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--draws',type=int,default=10000);a=p.parse_args()
b=pd.read_csv(a.baseline);e=pd.read_csv(a.experiment);keys=['series_index','video_id','start_frame','label']
pd.testing.assert_frame_equal(b[keys],e[keys]);rows=[]
for metric in ['ADE','FDE']:
 d=e[metric].to_numpy()-b[metric].to_numpy();rng=np.random.default_rng(a.seed);groups=[d[b.video_id.to_numpy()==v] for v in b.video_id.unique()];iid=[];cluster=[]
 for _ in range(a.draws):
  iid.append(float(rng.choice(d,len(d),replace=True).mean()));cluster.append(float(np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]).mean()))
 q=np.quantile(iid,[.025,.975]);c=np.quantile(cluster,[.025,.975]);rows.append(dict(metric=metric,n=len(d),n_videos=len(groups),mean_difference=float(d.mean()),wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d) else 1.,iid_95_low=q[0],iid_95_high=q[1],video_cluster_95_low=c[0],video_cluster_95_high=c[1],seed=a.seed,draws=a.draws))
with Path(a.output).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
