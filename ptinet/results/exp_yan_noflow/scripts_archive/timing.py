"""Step 2, real validation inputs, synchronized forward and official RAFT."""
import sys
sys.dont_write_bytecode=True
import json,time,subprocess,os,platform,gc
from pathlib import Path
import common as c

def main():
 started=time.monotonic();version=c.setup();import torch,numpy as np,pandas as pd
 out=c.OUT/'step2_timing';out.mkdir(exist_ok=True)
 # One full dataset pass provides the same 110 true input windows for every model.
 d=c.data(c.cfg());loader=torch.utils.data.DataLoader(torch.utils.data.Subset(d,range(110)),batch_size=1,num_workers=8,shuffle=False)
 nets={};counts={};measurements={};modes=[('Baseline',True,True),('光フローなし',True,False),('画像なし',False,True)]
 for name,image,flow in modes:
  torch.manual_seed(0);net=c.model(c.cfg(use_image=image,use_opticalflow=flow));net.eval();nets[name]=net
  counts[name]=sum(p.numel() for p in net.parameters());measurements[name]=[]
 torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  for index,b in enumerate(loader):
   t=c.inputs(b)
   # Rotate order to reduce persistent model-order bias; all inputs are prepared before timing.
   names=list(nets);names=names[index%3:]+names[:index%3]
   for name in names:
    torch.cuda.synchronize();start=time.perf_counter();output=c.forward(nets[name],t);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
    if index>=10:measurements[name].append(elapsed)
    del output
   del t,b
 del nets;gc.collect();torch.cuda.empty_cache()
 # Official RAFT worker from A is reused without altering its source; globals OUT redirected in a fresh process.
 command="import sys;sys.dont_write_bytecode=True;from pathlib import Path;sys.path.insert(0,'LOCAL_EXPERIMENT/PTINet/tools');import profile_val;p=Path('LOCAL_EXPERIMENT/reports/flow_study/step2_timing');profile_val.OUT=p;profile_val.raft()"
 subprocess.run([sys.executable,'-c',command],check=True,timeout=1800)
 raft=json.loads((out/'raft_result.json').read_text())['resolutions'];raft_ms={(r['width'],r['height']):r['median_ms'] for r in raft}
 base_ms=float(np.median(measurements['Baseline']));end_base=base_ms+raft_ms[(960,540)]
 rows=[]
 for name,image,flow in modes:
  vals=measurements[name];model_ms=float(np.median(vals));raft_cost=raft_ms[(960,540)] if flow else 0.;total=model_ms+raft_cost
  rows.append({'variant':name,'model_median_ms':model_ms,'model_p90_ms':float(np.percentile(vals,90)),'raft_median_ms':raft_cost,'end_to_end_estimate_ms':total,'model_fps':1000/model_ms,'end_to_end_fps':1000/total,'parameters':counts[name],'model_reduction_ms':base_ms-model_ms,'model_reduction_percent':100*(base_ms-model_ms)/base_ms,'end_to_end_reduction_percent':100*(end_base-total)/end_base,'commit_id':version})
 row=rows[0].copy();row.update(variant='Baseline・RAFT低解像度',raft_median_ms=raft_ms[(426,240)],end_to_end_estimate_ms=base_ms+raft_ms[(426,240)],end_to_end_fps=1000/(base_ms+raft_ms[(426,240)]),end_to_end_reduction_percent=100*(end_base-base_ms-raft_ms[(426,240)])/end_base);rows.append(row)
 pd.DataFrame(rows).to_csv(out/'summary.csv',index=False,float_format='%.17g')
 pd.DataFrame([{'variant':name,'sample_index':i+10,'ms':v} for name,values in measurements.items() for i,v in enumerate(values)]).to_csv(out/'raw_timings.csv',index=False)
 c.plot_setup();import matplotlib.pyplot as plt
 for field,title,file,selected in [('model_median_ms','モデル推論時間（前処理・RAFTを除く）','model_latency.png',rows[:2]),('end_to_end_estimate_ms','エンドツーエンド推定（RAFTを含む）','end_to_end_latency.png',rows[:2]),('end_to_end_estimate_ms','入力分岐とRAFT解像度の処理時間','all_variants.png',rows)]:
  fig,ax=plt.subplots(figsize=(9,5));values=[r[field] for r in selected];bars=ax.bar([r['variant'] for r in selected],values,color=['#2878a5','#d77524','#52914b','#8a6fae'][:len(selected)])
  for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.02,f'{value:.1f} ms',ha='center')
  if len(selected)==2:
   ax.axhline(100,color='#666666',linestyle='--',label='100 ms（10 fps）');ax.axhline(33.3,color='#a04b4b',linestyle=':',label='33.3 ms（30 fps）');ax.legend()
  ax.set(ylabel='処理時間 [ms]',title=title,ylim=(0,max(max(values)*1.2,110)));fig.tight_layout();fig.savefig(out/file,dpi=180);plt.close(fig)
 machine={'computer':platform.node(),'system':platform.platform(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':torch.version.hip,'batch':1,'warmup':10,'measured':100,'cpu_model':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown')}
 c.atomic(out/'machine.json',machine)
 c.readme(out,'推論時間の実測',f'コマンド: `{c.PYTHON} {Path(__file__).resolve()}`。所要時間{time.monotonic()-started:.1f}秒。\n\n計測条件: {machine}。同じval100系列（10series warm-up）、batch1、eval/no_grad、cuda synchronize前後、中央値とP90。全モデルが横断と軌跡の両出力を返すまで。実験checkpointは読み込まず、作者constructorのResNet ImageNet初期化を保持し、その他はseed0初期化。入力テンソルの準備、読込、転送、RAFTはモデル推論時間に含まない。順番は各サンプルで回転。\n\nRAFTは公式raft-things.pth、FP32、iters20、各解像度5回warm-upと50対val隣接フレーム。実測はGPU推論のみ。960×540は960×544、426×240は432×240にpad。前段Aの再利用workerで当日再計測、旧結果の流用ではない。\n\n主指標はモデル推論時間。補足の端到端推定は1frameあたりRAFT1回＋観測窓全体forward1回の中央値和で、IO等は含まずオフライン事前計算時にはRAFT不要。モデルパラメータ数は重み総要素数、未使用attrib層を含め実モデル全体を数える。\n\n本体画像の既定リサイズはPIL W240×H426のまま。attrib_decoder / fc_attrib 有定义，但 forward 中未使用。')
 c.atomic(out/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,'commit_id':version})
 print('TIMING_COMPLETE',flush=True)
if __name__=='__main__':main()
