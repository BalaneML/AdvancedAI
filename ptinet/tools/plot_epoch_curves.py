"""Redraw published epoch CSV, CPU only. No training records or datasets needed."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=pd.read_csv(a.csv);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
font=Path('/usr/share/fonts/truetype/fonts-japanese-gothic.ttf')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
print('Columns',list(d))
for metric in ['ADE','FDE']:
 fig,ax=plt.subplots(figsize=(9,4))
 for model,label in [('D1','フローあり'),('noflow','フローなし')]:
  key=model+'_val_'+metric;ax.plot(d.epoch_number,d[key],label=label)
  best=d[model+'_val_ADE'].idxmin();ax.scatter(d.loc[best,'epoch_number'],d.loc[best,key],marker='*',s=160)
 ax.set(xlabel='エポック',ylabel='validation '+metric+' [px]');ax.legend();fig.tight_layout();fig.savefig(out/('val_'+metric.lower()+'_curves.png'),dpi=160);plt.close(fig)
