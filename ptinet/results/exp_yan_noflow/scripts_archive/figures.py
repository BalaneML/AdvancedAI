"""Step 9: Japanese scientific tables/figures and a local report-only ZIP."""
import sys
sys.dont_write_bytecode=True
import json,time,shutil,zipfile
from pathlib import Path
import common as c

def main():
 plan=json.loads((c.OUT/'plan.json').read_text())
 titan_skipped=plan.get('titan_data_steps')=='skipped_by_user_this_run'
 start=time.monotonic();version=c.setup(gpu=False)
 import numpy as np,pandas as pd
 c.plot_setup();import matplotlib.pyplot as plt
 out=c.OUT/'step9_figures';out.mkdir(exist_ok=True)
 for source,name in [('step2_timing/model_latency.png','01_model_latency.png'),('step2_timing/end_to_end_latency.png','02_end_to_end_latency.png'),('step2_timing/all_variants.png','03_all_latency.png'),('step6_images/rgb_flow_comparison.png','04_rgb_flow.png')]:
  p=c.OUT/source
  if p.exists():shutil.copy2(p,out/name)
 tables=[];time_path=c.OUT/'step2_timing/summary.csv';timing=pd.read_csv(time_path) if time_path.exists() else None
 if timing is not None:
  renamed=timing.rename(columns={'variant':'構成','model_median_ms':'モデル推論時間_ms','model_p90_ms':'P90_ms','raft_median_ms':'RAFT_ms','end_to_end_estimate_ms':'端到端推定_ms','model_fps':'モデル_fps','end_to_end_fps':'端到端_fps','parameters':'パラメータ数','model_reduction_ms':'短縮_ms','model_reduction_percent':'短縮率_百分率','end_to_end_reduction_percent':'端到端短縮率_百分率','commit_id':'Commit_ID'})
  renamed.to_csv(out/'処理時間比較.csv',index=False,encoding='utf-8-sig')
 p=c.OUT/'step4_jaad_noflow/summary.csv'
 if p.exists():
  jaad=pd.read_csv(p)
  for label,r in zip(['Baseline（D1）','光フローなし'],jaad.to_dict('records')):
   ms=None
   if timing is not None:
    match=timing[timing.variant==('Baseline' if label.startswith('Baseline') else '光フローなし')]
    if len(match):ms=float(match.iloc[0].model_median_ms)
   tables.append({'データセット':'JAAD・完全','構成':label,'ADE_px':r['ADE'],'FDE_px':r['FDE'],'F1':r['F1'],'正解率':r['Accuracy'],'PR_AUC':r['PR_AUC'],'モデル推論時間_ms':ms,'Commit_ID':r['commit_id'],'状態':'完了'})
  for metric in ['ADE','FDE']:
   fig,ax=plt.subplots(figsize=(7,4.5));values=jaad[metric].to_numpy();bars=ax.bar(['Baseline（D1）','光フローなし'],values,color=['#2878a5','#d77524'])
   for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.02,f'{v:.3f}',ha='center')
   ax.set(ylabel=metric+' [pixel]',title='JAAD完全設定・validation',ylim=(0,max(values)*1.2));fig.tight_layout();fig.savefig(out/f'JAAD_{metric}.png',dpi=180);plt.close(fig)
 titan=c.OUT/'step8_titan/summary.csv'
 if titan.exists() and 'ADE' in pd.read_csv(titan,nrows=0).columns:
  tit=pd.read_csv(titan)
  for row in tit.to_dict('records'):tables.append({'データセット':'TITAN・縮小',**row})
 else:
  for label in ['Baseline','光フローなし']:tables.append({'データセット':'TITAN・縮小','構成':label,'状態':('本回見送り（ユーザー指定）' if titan_skipped else '未実施：データ待ち'),'Commit_ID':version})
 if tables:pd.DataFrame(tables).to_csv(out/'精度比較.csv',index=False,encoding='utf-8-sig')
 for step in ['step3_masking','step4_jaad_noflow']:
  p=c.OUT/step/'groups.csv'
  if p.exists():
   grouped=pd.read_csv(p)
   grouped['condition']=grouped['condition'].replace({'base':'Baseline','zero_flow':'光フロー入力ゼロ','zero_image':'画像入力ゼロ','D1':'Baseline（D1）','noflow':'光フローなし'})
   grouped['stratum']=grouped['stratum'].astype(str).replace({'all':'全体','0':'非横断','1':'横断','moving':'移動','stationary':'静止','stopped':'停止','unknown_or_mixed':'不明・混在'})
   grouped['group']=grouped['group'].replace({'all':'全体','label':'横断ラベル','pedestrian_motion':'歩行者動静','ego_motion':'自車動静'})
   grouped.rename(columns={'condition':'条件','group':'分割軸','stratum':'群','N':'系列数','commit_id':'Commit_ID'}).to_csv(out/(step+'_場面別.csv'),index=False,encoding='utf-8-sig')
 p=c.OUT/'step5_information';frames=[pd.read_csv(x) for x in sorted(p.glob('*_summary.csv'))]
 if frames:
  small=pd.concat(frames,ignore_index=True);small=small.assign(_order=small.run.str.contains('trajectory_flow').astype(int)).sort_values('_order').drop(columns='_order');small.to_csv(p/'summary.csv',index=False,float_format='%.17g')
  labels=['軌跡＋光フロー' if 'trajectory_flow' in name else '軌跡のみ' for name in small.run]
  export=small[['run','ADE','FDE','F1','Accuracy','PR_AUC','commit_id']].copy();export['構成']=labels;export.to_csv(out/'光フロー追加情報.csv',index=False,encoding='utf-8-sig')
  fig,axes=plt.subplots(1,2,figsize=(10,4.5))
  for ax,metric in zip(axes,['ADE','FDE']):
   values=small[metric].to_numpy();bars=ax.bar(labels,values,color=['#2878a5','#d77524'][:len(values)])
   for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.02,f'{v:.3f}',ha='center')
   ax.set(ylabel=metric+' [pixel]',ylim=(0,max(values)*1.2))
  fig.suptitle('縮小設定・光フローの追加情報');fig.tight_layout();fig.savefig(out/'05_trajectory_flow.png',dpi=180);plt.close(fig)
  c.readme(p,'軌跡のみ vs 軌跡＋光フロー', 'summary.csvに2モデルの実測結果を併記。両方seed0・22epoch・train先頭25%動画・val全量、LCF/画像を使用しない。各runのconfig/complete.jsonに所要時間を記録。光フロー追加で改善すれば本学習条件で追加情報を利用した証拠となるが、差が小さくても光フローに情報が全くないことの証明にはならない。画像座標と小規模学習の制約を残す。')
 c.readme(out,'発表用図表','コマンド: `'+c.PYTHON+' '+str(Path(__file__).resolve())+'`。主指標は事前計算済光フローを前提とするモデル推論時間、補足のみRAFTを含む。図はPNG、表はUTF-8 BOM CSV、日本語ラベル。未実施のTITAN結果は空欄として明記し、偽の平均・SDを作らない。各ステップの詳細と限界は元README、対応検定はstep4/paired_test.csv。')
 # Global durable conclusions are assembled only from completed, inspectable evidence.
 lines=['# 光流验证全套：结果与执行状态','',f'更新时间：{c.now()}。Commit `{version}`。仅validation，test未读取/评价。TITAN个人许可数据限本机，任何下载链接均不写入本包。','', '## 已得到的结果','']
 if timing is not None:
  a=timing[timing.variant=='Baseline'].iloc[0];b=timing[timing.variant=='光フローなし'].iloc[0]
  lines.append(f"- 第2步：模型推论时间（不含RAFT）Baseline {a.model_median_ms:.3f}ms，无光流 {b.model_median_ms:.3f}ms，缩短{b.model_reduction_percent:.2f}%；含RAFT在线估计分别{a.end_to_end_estimate_ms:.3f}/{b.end_to_end_estimate_ms:.3f}ms。")
 if (c.OUT/'step4_jaad_noflow/summary.csv').exists():
  f=pd.read_csv(c.OUT/'step4_jaad_noflow/summary.csv');a,b=f.iloc[0],f.iloc[1]
  lines.append(f'- 第4步：D1 ADE/FDE={a.ADE:.6f}/{a.FDE:.6f}，去光流={b.ADE:.6f}/{b.FDE:.6f}。差值与95%CI、Wilcoxon p值见paired_test.csv。不预设精度不变，未指定非劣性界限，因此不把不显著自动解释为等价。')
 if (c.OUT/'step3_masking/summary.csv').exists() and len(pd.read_csv(c.OUT/'step3_masking/summary.csv'))==3:lines.append('- 第3步：D1修正版baseline/光流置零/图像置零及横穿、行人动静、自车动作分组均在step3_masking；与结构去分支的重训练区分。')
 if (c.OUT/'step6_images/summary.csv').exists():lines.append('- 第6步：真实JAAD RGB/flow并排图与近白像素统计已完成；白色可视化不等于绝对零运动。')
 if frames:lines.append('- 第5步：轨迹与轨迹＋光流的小设定结果在step5_information/summary.csv及step9图表；只对当前规模、seed0作判断。')
 lines+=['','## 未完成、偏差和复现说明','', '- TITAN下载及第7/8步：用户明确要求本轮跳过。授权已过期，本轮没有使用真实TITAN数据。titan_code_prepare包含CPU合成接口验证和未来真实数据入口，不是TITAN精度结果。' , '- 第8.5步：用户已在任何对应训练启动前取消；不运行、不把它计为已完成。','- 缩小尺寸的最终选择、训练视频列表、22epoch预算见plan.json；默认全尺寸不改宽高顺序。','- 作者指定的hpa重复、ConvLSTM5帧和RGBA可视化流不修正。baseline默认输出一致性、8组合smoke、disabled branch不读文件等在step1_prepare。','- D1训练早于git建立；报告中将其明确标为pre-git并附当时代码snapshot，不能伪造当时训练commit。本次训练及每次评价前要求当前repo clean，实际commit写入summary。','- 首次调度脚本与Python标准库queue同名，在进入GPU任务前导入失败，已更名为orchestrate；首次错误日志保留，不修改模型以绕过该错误。','- 模型源码仅本机git，无remote/push。各step包含命令、用时、差异。状态以status.json/日志为准，错误列于各queue_result.json。','', '## 文件','', '- step1～step9目录：README、summary.csv、图片及本机原始证据。','- step9_figures：可用于PPT的日本语PNG/CSV。','- 最终ZIP只含README、汇总/分组/检验表、PNG和代码diff，不含数据集、checkpoint或TITAN原始样本。','']
 (c.OUT/'README.md').write_text('\n'.join(lines))
 rows=[{'step':p.name,'README':(p/'README.md').exists(),'summary':(p/'summary.csv').exists(),'commit_id':version} for p in sorted(c.OUT.glob('step*')) if p.is_dir()];pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
 package=c.ROOT/'reports'/'PTINet_flow_study_results.zip';temporary=package.with_suffix('.tmp.zip')
 with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for path in sorted(c.OUT.rglob('*')):
   if not path.is_file() or any(x.startswith('.') for x in path.relative_to(c.OUT).parts):continue
   if path.suffix.lower() in ['.md','.png','.diff'] or path.name in ['summary.csv','groups.csv','paired_test.csv','coverage.csv'] or (path.parent==out and path.suffix=='.csv'):
    z.write(path,Path('flow_study')/path.relative_to(c.OUT))
 with zipfile.ZipFile(temporary) as z:assert z.testzip() is None
 temporary.replace(package)
 c.atomic(out/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-start,'commit_id':version,'zip':str(package)})
 print('FIGURES_AND_ZIP_COMPLETE',package,flush=True)
if __name__=='__main__':main()
