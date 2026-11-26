"""CPU-only Step 6; inspect genuine JAAD validation RGB and existing RGBA flow."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HIP_VISIBLE_DEVICES']='';os.environ['ROCR_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
import time,json
from pathlib import Path
import common as c

def main():
 start=time.monotonic();version=c.setup(gpu=False)
 import numpy as np,pandas as pd
 from PIL import Image
 out=c.OUT/'step6_images';out.mkdir(exist_ok=True)
 dataset=c.data(c.cfg(),images=False,flow=False);records=[]
 for i in range(len(dataset)):
  info=c.meta(dataset,i);info['label']=int(dataset.data.iloc[i].label);records.append(info)
 frame=pd.DataFrame(records);frame.to_csv(out/'val_scene_metadata.csv',index=False)
 selected=[];used=set();coverage=[]
 for ego in ['moving','stopped']:
  for ped in ['moving','stationary']:
   candidates=frame[(frame.ego_motion==ego)&(frame.pedestrian_motion==ped)]
   chosen=[]
   for row in candidates.to_dict('records'):
    if row['video_id'] not in used:chosen.append(row);used.add(row['video_id'])
    if len(chosen)==2:break
   if len(chosen)<2:
    for row in candidates.to_dict('records'):
     if row['series_index'] not in {x['series_index'] for x in chosen}:chosen.append(row)
     if len(chosen)==2:break
   selected+=chosen;coverage.append({'ego':ego,'pedestrian':ped,'available_series':len(candidates),'selected':len(chosen)})
 c.plot_setup();import matplotlib.pyplot as plt
 fig,axes=plt.subplots(max(1,len(selected)),2,figsize=(12,max(3,2.8*len(selected))),squeeze=False)
 summaries=[]
 for j,row in enumerate(selected):
  rgb_path=Path(row['rgb_path']);flow_path=Path(str(rgb_path).replace('/images/','/opticalflow/'))
  with Image.open(rgb_path) as im:rgb=np.asarray(im.convert('RGB'))
  with Image.open(flow_path) as im:
   rgba=np.asarray(im);assert im.mode=='RGBA' and im.size==(960,540)
  color=rgba[:,:,:3];white=(color.min(axis=2)>=245)&((color.max(axis=2)-color.min(axis=2))<=10)
  value=float(white.mean());summaries.append({**row,'flow_path':str(flow_path),'near_white_ratio':value,'RGBA_alpha_all_255':bool((rgba[:,:,3]==255).all()),'commit_id':version})
  axes[j,0].imshow(rgb);axes[j,1].imshow(color)
  from matplotlib.patches import Rectangle
  cx,cy,bw,bh=dataset.data.iloc[row['series_index']].bounding_box[0]
  axes[j,0].add_patch(Rectangle((cx-bw/2,cy-bh/2),bw,bh,fill=False,edgecolor='#00c060',linewidth=2))
  ego='走行' if row['ego_motion']=='moving' else '停止';ped='移動' if row['pedestrian_motion']=='moving' else '静止'
  axes[j,0].set_title(f"{row['video_id']}・観測区間：自車{ego}／歩行者{ped}")
  axes[j,1].set_title(f'光フロー・白色比率 {value:.1%}')
  for ax in axes[j]:ax.axis('off')
 fig.tight_layout();fig.savefig(out/'rgb_flow_comparison.png',dpi=150);plt.close(fig)
 pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False,float_format='%.17g');pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
 c.readme(out,'光フローPNGの確認',f'コマンド: `{c.PYTHON} {Path(__file__).resolve()}`（CPUのみ）。所要時間{time.monotonic()-start:.1f}秒。valから自車moving/stopped×歩行者moving/stationaryを各2例、可能な限り異なる動画を選ぶ。実際の群カバレッジはcoverage.csv。緑枠は対象歩行者の観測初frame bbox、動静ラベルは観測区間に基づく。\n\n白色判定：RGB全channel≥245かつmax−min≤10、alphaは比率計算から除外。RGBA alpha=255も確認。flow_vizはframeごとの相対正規化なので、白色比率は小さな可視化変位の指標で、絶対速度や静止の証明ではない。\n\n実装は公式RAFT FP32 iters20→flow_viz色画像→RGBA alpha255 PNG（960×540）。RAFTは960×540入力をpadし、出力をunpadして同解像度で保存、DataLoaderで4channelとして読み込み既定PIL W240×H426にresizeする。元のwidth/height順序は修正しない。\n\n指令に記載された論文設定：RAFT、240×420。本実装との相違は保存/計算解像度960×540、モデル入力の実形状H426×W240、4channel可視化PNG。論文の生flow channel数/正規化等は提示資料から未確認であり推測しない。')
 c.atomic(out/'status.json',{'status':'complete','commit_id':version,'scenes':len(selected),'elapsed_seconds':time.monotonic()-start})
 print('IMAGES_COMPLETE',len(selected),flush=True)
if __name__=='__main__':main()
