"""Step 3, Corrected D1 paired masking, full val, no visual cache."""
import sys
sys.dont_write_bytecode=True
import time,json
from pathlib import Path
import common as c

def main():
 start=time.monotonic();version=c.setup();import pandas as pd
 out=c.OUT/'step3_masking';out.mkdir(exist_ok=True);cp=c.ROOT/'runs/jaad_corr_lcf_out15_s0/model_best0.001_512.pkl'
 rows=[];groups=[]
 for condition in ['base','zero_flow','zero_image']:
  f,row=c.evaluate(c.cfg(),cp,out/condition,condition);rows.append(row)
  for g in c.grouped(f):groups.append({'condition':condition,**g,'commit_id':version})
  pd.DataFrame(rows).to_csv(out/'summary.csv',index=False,float_format='%.17g');pd.DataFrame(groups).to_csv(out/'groups.csv',index=False,float_format='%.17g')
  print('MASK_COMPLETE',condition,flush=True)
 original=pd.read_csv(c.ROOT/'reports/corrected_val/jaad_corr_lcf_out15_s0/trajectory_samples.csv',float_precision='round_trip')
 base=pd.read_csv(out/'base/samples.csv',float_precision='round_trip')
 import numpy as np
 difference=max(float(np.abs(base.ADE-original.ADE).max()),float(np.abs(base.FDE-original.FDE).max()))
 assert difference<1e-5,('D1 baseline changed',difference)
 c.atomic(out/'baseline_reproduction.json',{'max_per_sample_error_vs_prior_D1':difference,'commit_id':version})
 c.readme(out,'Corrected D1・場面別マスキング',f'コマンド: `{c.PYTHON} {Path(__file__).resolve()}`。所要時間{time.monotonic()-start:.1f}秒。各条件seed0をモデル生成前にreset、batch2、workers8、全val616系列。各条件全forward再計算、マスキングは学習済分岐を残したまま入力を0にする操作で、分岐を除く再学習と区別する。\n\nADE/FDEは中心のpixel距離（原utils.speed2posのclipを保持）。歩行者の動静は観測期間のbbox中心の始点終点距離>5pixelをmovingとするため、カメラ運動や追跡ノイズも含む画像座標上の指標であり真の世界座標速度ではない。\n\n自車はJAAD annotations_vehicleの各観測frameに対応するactionを利用。moving_fast/moving_slow/accelerating/deceleratingが過半数ならmoving、stoppedが過半数ならstopped、それ以外はunknown_or_mixedとして除外せず別群に残す。横断labelはfuture cross_label。群は重複するため各行を足し上げない。\n\n過去D1 valとの最大逐系列ADE/FDE差は{difference:.9g}。D1訓練時はgit導入前であり、訓練commitを遡及して捏造しない。本表commitは今回評価のコード。')
 c.atomic(out/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-start,'commit_id':version})
if __name__=='__main__':main()
