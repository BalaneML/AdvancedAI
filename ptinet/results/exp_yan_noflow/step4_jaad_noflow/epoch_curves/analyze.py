"""CPU-only analysis of existing epoch histories. No torch, training or evaluation."""
import os
for key in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:os.environ[key]=''
import sys
sys.dont_write_bytecode=True
import json,re,hashlib,subprocess,zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('LOCAL_EXPERIMENT');OUT=Path(__file__).resolve().parent;PROJECT=ROOT/'PTINet_corrected'
os.environ['MPLCONFIGDIR']=str(OUT/'.matplotlib_cache')
START=time.monotonic()
RUNS={'D1':'jaad_corr_lcf_out15_s0','noflow':'jaad_corr_noflow_out15_s0'}
LOGS={'D1':ROOT/'logs/jaad_corr_lcf_out15_s0.log','noflow':ROOT/'reports/flow_study/step4_jaad_noflow/train_job.log'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 frames={};audit={};sources=[]
 for key,name in RUNS.items():
  path=ROOT/'runs'/name/'data_final0.001_512.csv';frame=pd.read_csv(path,float_precision='round_trip')
  assert len(frame)==22 and np.array_equal(frame.epoch.to_numpy(),np.arange(22))
  assert np.isfinite(frame[['ade','fde']]).all().all()
  # Independent source: original stdout records ADE/FDE rounded to 4 decimals.
  records=re.findall(r'^e:\s*(\d+)\s*\| ade:\s*([-+\d.eE]+)\s*\| fde:\s*([-+\d.eE]+)',LOGS[key].read_text(),re.M)
  assert len(records)==22
  rounded=pd.DataFrame(records,columns=['epoch','ade','fde']).astype({'epoch':int,'ade':float,'fde':float})
  assert np.array_equal(rounded.epoch,frame.epoch)
  errors=np.abs(rounded[['ade','fde']].to_numpy()-frame[['ade','fde']].to_numpy())
  assert errors.max()<=.00005001
  frames[key]=frame
  audit[key]={'full_precision_source':str(path),'sha256':sha(path),'log_source':str(LOGS[key]),'log_sha256':sha(LOGS[key]),'epochs':22,'max_difference_vs_rounded_log':float(errors.max())}
  sources.extend([path,LOGS[key]])
 combined=pd.DataFrame({'epoch':np.arange(22),'epoch_number':np.arange(1,23),'D1_val_ADE':frames['D1'].ade,'D1_val_FDE':frames['D1'].fde,'noflow_val_ADE':frames['noflow'].ade,'noflow_val_FDE':frames['noflow'].fde})
 best={k:int(f.ade.idxmin()) for k,f in frames.items()}
 # train.py saves best only on strict improvement, so the first minimum is selected.
 combined['D1_selected_best']=combined.epoch==best['D1'];combined['noflow_selected_best']=combined.epoch==best['noflow']
 combined['noflow_ADE_minus_D1']=combined.noflow_val_ADE-combined.D1_val_ADE
 combined['noflow_ADE_le_D1']=combined.noflow_val_ADE<=combined.D1_val_ADE
 combined.to_csv(OUT/'epoch_metrics.csv',index=False,float_format='%.17g')
 summaries=[]
 for key,f in frames.items():
  values=f.ade.to_numpy();differences=np.diff(values)
  summaries.append({'model':key,'n_epochs':len(f),'selected_best_epoch_index':best[key],'selected_best_epoch_number':best[key]+1,'best_val_ADE':float(f.ade.iloc[best[key]]),'FDE_at_ADE_selected_best':float(f.fde.iloc[best[key]]),'last5_mean_ADE':float(f.ade.iloc[-5:].mean()),'all_epochs_mean_ADE':float(f.ade.mean()),'all_epochs_ADE_sample_std_ddof1':float(f.ade.std(ddof=1)),'all_epochs_ADE_min':float(f.ade.min()),'all_epochs_ADE_max':float(f.ade.max()),'max_abs_adjacent_epoch_ADE_change':float(np.abs(differences).max()),'mean_abs_adjacent_epoch_ADE_change':float(np.abs(differences).mean()),'adjacent_ADE_changes_sample_std_ddof1':float(np.std(differences,ddof=1))})
 pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False,float_format='%.17g')
 count=int(combined.noflow_ADE_le_D1.sum())
 comparison={'noflow_ADE_le_D1_epochs':count,'total_epochs':22,'epoch_numbers_noflow_ADE_le_D1':combined.loc[combined.noflow_ADE_le_D1,'epoch_number'].tolist(),'last5_mean_ADE_difference_noflow_minus_D1':float(frames['noflow'].ade.iloc[-5:].mean()-frames['D1'].ade.iloc[-5:].mean()),'selected_best_rule':'first strict minimum of training-time val ADE; FDE marker uses the same checkpoint epoch','no_training_or_inference':True,'validation_only':True}
 (OUT/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
 baseline=(ROOT/'reports/flow_study/step1_prepare/baseline_commit.txt').read_text().strip()
 launch=json.loads((ROOT/'runs'/RUNS['noflow']/'launch.json').read_text());version=launch['commit_id']
 codes={k:subprocess.check_output(['git','-C',str(PROJECT),'show',commit+':train.py'],text=True) for k,commit in [('D1',baseline),('noflow',version)]}
 snap=ROOT/'reports/corrected_val'/RUNS['D1']/'intention'/f"{RUNS['D1']}_protected_before.json"
 saved=json.loads(snap.read_text());assert hashlib.sha256(codes['D1'].encode()).hexdigest()==saved['train.py']['sha256']
 assert 'torch.manual_seed(0)' in codes['D1'] and 'torch.manual_seed(args.seed)' in codes['noflow'] and launch['config']['seed']==0
 for code in codes.values():
  assert code.index('net.eval()')<code.index('for idx, val_in in enumerate(dataloader_val):')<code.index('        net.train()')
  assert 'if ade < best_ade:' in code and 'with torch.no_grad():' in code
 cfg=json.loads((ROOT/'runs'/RUNS['D1']/'run_telemetry.json').read_text())['config']
 common_fields=['dataset','input','output','stride','skip','batch_size','n_epochs','lr','hidden_size','use_image','image_network','use_attribute','loader_workers','pin_memory','from_file']
 assert all(cfg[key]==launch['config'][key] for key in common_fields)
 audit['comparability']={'D1_seed':0,'noflow_seed':0,'both_validation_eval':True,'both_validation_no_grad':True,'both_training_resume_train_after_val':True,'D1_code_snapshot':baseline,'noflow_training_commit':version,'D1_snapshot_matches_saved_training_source_sha256':True,'same_critical_settings':common_fields,'intentional_difference':'use_opticalflow True vs False','D1_trained_before_git':True}
 (OUT/'source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import font_manager,rcParams
 font='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf';font_manager.fontManager.addfont(font);rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();rcParams['axes.unicode_minus']=False
 import matplotlib.pyplot as plt
 for metric in ['ADE','FDE']:
  fig,ax=plt.subplots(figsize=(10,5.5))
  for key,label,color,marker in [('D1','D1（光フローあり）','#2878a5','o'),('noflow','光フローなし','#d77524','s')]:
   x=frames[key].epoch.to_numpy()+1;y=frames[key][metric.lower()].to_numpy();b=best[key]
   ax.plot(x,y,color=color,marker=marker,markersize=3.5,label=label,linewidth=1.7)
   ax.scatter([b+1],[y[b]],marker='*',s=220,color=color,edgecolor='black',linewidth=.6,zorder=5)
   ax.text(.025,.95 if key=='D1' else .82,f'{label}\n選択best: epoch {b+1}, {metric}={y[b]:.3f}',transform=ax.transAxes,fontsize=9,color=color,va='top',bbox={'facecolor':'white','alpha':.9,'edgecolor':color,'boxstyle':'round,pad=.4'})
  ax.set(xlabel='エポック（1始まり）',ylabel=f'validation {metric} [pixel]',title=f'各epochのvalidation {metric}（bestはADE最小で選択）')
  ax.set_xticks(np.arange(1,23));ax.set_xlim(.5,22.5);ax.grid(alpha=.22);ax.legend(loc='upper right');fig.tight_layout();fig.savefig(OUT/f'val_{metric.lower()}_curves.png',dpi=180);plt.close(fig)
 rows=pd.DataFrame(summaries)
 lines=['# D1 与去光流：每轮 validation 曲线','',f'运行命令：`{sys.executable} {Path(__file__).resolve()}`。纯CPU读取已有记录，不导入torch、不训练、不加载checkpoint推理、不读取test。','', '## 数据来源和核对','', '- 两边均找到各22轮完整、未四舍五入的data_final0.001_512.csv（路径/哈希见source_audit.json）。epochs.json主要保存时间，不把它误当ADE来源。', '- 每轮ADE/FDE都与各自stdout日志的4位小数交叉核对，误差≤0.00005001。无需TensorBoard，也没有补跑任何评价。', '- epoch_metrics.csv：epoch是原始0起始索引，epoch_number是图中的1起始轮次；包含每轮两边ADE/FDE、best标记和差值。', '', '## 记录方式是否可比','', '- 两边都在每次validation前net.eval()，在torch.no_grad()下评价，结束后恢复net.train()。D1不是此前Original那种未切eval的训练。', '- D1训练源码从本机Baseline git快照恢复，并与D1诊断当时保存的train.py SHA256核对一致；D1原训练发生在git建立前，不伪造其历史commit。noflow使用launch.json记录的训练commit。', '- D1在源码硬编码torch.manual_seed(0)，noflow使用args.seed=0；关键训练/序列/加载器设置一致，都是全JAAD train和val、batch2、workers8、22epoch、input16/output15/stride16、原PIL W240×H426。', '- 结构差异为use_opticalflow；此外noflow版本增加的尺寸/视频子集开关均保持默认全量值，改进的batch变量在Baseline全输入时已通过逐输出完全一致性检查。', '- 同seed不意味着两种架构的随机初始化、训练样本次序或VAE采样流逐项完全相同：移除模块会改变随机数消耗。这是独立训练的单seed对照，不是同一模型每轮的配对干预。VAE在eval仍随机采样；训练日志每轮val还继承当时训练后的RNG状态。', '', '## 简单统计','',f'- 去光流ADE ≤ D1：**{count}/22轮**；1起始轮次={comparison["epoch_numbers_noflow_ADE_le_D1"]}。','', '|模型|best轮次（1起始）|best val ADE|最后5轮平均ADE|全22轮ADE样本标准差|ADE最大值|最大相邻轮绝对变化|','|---|---:|---:|---:|---:|---:|---:|']
 for row in summaries:lines.append('|'+row['model']+'|'+'|'.join(f'{row[key]:.6f}' if isinstance(row[key],float) else str(row[key]) for key in ['selected_best_epoch_number','best_val_ADE','last5_mean_ADE','all_epochs_ADE_sample_std_ddof1','all_epochs_ADE_max','max_abs_adjacent_epoch_ADE_change'])+'|')
 lines+=['', '样本标准差使用ddof=1；“最大值”同时提供全轮ADE最大值和最大相邻轮变化，以避免把水平与抖动混为一谈。summary.csv另含相邻差分标准差、平均绝对差分和全轮均值。最后5轮指1起始第18–22轮。','', '## best标记与解释边界','', '- 训练代码仅当ade < best_ade时保存best，故采用val ADE首次严格最小的轮次。FDE图的星号也标同一个被实际选中的checkpoint，不另挑FDE最小轮次。', '- 这些是训练过程中记录的每轮val值，与后来固定seed重新推理最佳checkpoint得到的12.1355/11.6186等数值不是同一条RNG轨迹，不能强行要求数值相等，也不能据此误判checkpoint选错。', '- 本分析可检查是否多个epoch、末5轮或总体抖动也支持同样趋势；仅两条单seed曲线不能证明已排除“偶然好checkpoint”或训练随机性。轮次相互相关，不把22轮视作22次独立训练或额外做伪配对显著性检验。','', '## 图表','', '![ADE](val_ade_curves.png)','', '![FDE](val_fde_curves.png)','']
 # Scoped, literal conclusion based on both selected minima and final-window behavior.
 d1,nf=summaries
 conclusion=f'去光流的训练日志best ADE={nf["best_val_ADE"]:.4f}，D1={d1["best_val_ADE"]:.4f}；末5轮平均ADE差（去光流−D1）={comparison["last5_mean_ADE_difference_noflow_minus_D1"]:+.4f}。结合胜出轮数{count}/22和标准差判断趋势，不只展示各自单个最低点。'
 lines.insert(lines.index('## best标记与解释边界'),conclusion+'\n\n**不能据此确认已排除偶然挑中好checkpoint。** 去光流虽在15/22轮的ADE不高于D1，但其最后5轮均值更差、全轮标准差和最大值更大；第18轮（1起始）的ADE尖峰65.9272明显影响末段均值，原样保留，不删除异常点。两边都有较大尖峰，因此单看best会掩盖训练不稳定性。\n')
 (OUT/'README.md').write_text('\n'.join(lines))
 assert sha(sources[0])==audit['D1']['sha256'] and sha(sources[2])==audit['noflow']['sha256']
 (OUT/'validation_checks.json').write_text(json.dumps({'status':'passed','epochs_each':22,'all_log_rows_crosschecked':True,'validation_mode_and_seed_verified':True,'best_epoch_rule_verified':True,'CPU_only':True,'re_evaluation':False,'read_test':False,'elapsed_seconds':time.monotonic()-START},indent=2))
 archive=OUT.parent/'PTINet_epoch_curves.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.iterdir()):
   if p.is_file():z.write(p,Path('epoch_curves')/p.name)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 print(json.dumps({'summary':summaries,'comparison':comparison,'zip':str(archive),'bytes':archive.stat().st_size},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
