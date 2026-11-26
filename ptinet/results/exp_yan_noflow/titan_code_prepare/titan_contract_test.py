"""CPU synthetic interface checks only. No real TITAN data/results; no GPU use."""
import os
for key in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:os.environ[key]=''
import sys
sys.dont_write_bytecode=True
import tempfile,json,time,gc
from pathlib import Path
import common as c

def main():
 start=time.monotonic();version=c.setup(gpu=False)
 import numpy as np,pandas as pd,torch
 from PIL import Image
 from types import SimpleNamespace
 from tools.make_fake_titan import make_clip
 from titan_prepare import prepare
 from datasets.titan import TITAN
 from model.network_image import PTINet
 out=c.OUT/'titan_code_prepare';out.mkdir(exist_ok=True);rows=[]
 with tempfile.TemporaryDirectory(prefix='ptinet_titan_contract_',dir='/tmp') as temp:
  temp=Path(temp);ann=temp/'annotations';ann.mkdir();spl=temp/'splits';spl.mkdir();images=temp/'frames';images.mkdir()
  rng=np.random.default_rng(0)
  for i in range(1,5):
   video=f'clip_{i}';frame=make_clip(rng,n_frames=90,n_peds=4);frame.to_csv(ann/(video+'.csv'),index=False)
   image_dir=images/video/'images';flow_dir=images/video/'opticalflow';image_dir.mkdir(parents=True);flow_dir.mkdir()
   for name in frame.frames.unique():
    Image.new('RGB',(96,64),(80,110,140)).save(image_dir/name)
    Image.new('RGBA',(96,64),(245,245,245,255)).save(flow_dir/name)
  (spl/'train_set.txt').write_text('clip_1\nclip_2\n');(spl/'val_set.txt').write_text('clip_3\nclip_4\n')
  # Deliberately no test_set.txt or test data. Any dependency on test would fail this test.
  processed=temp/'processed';audit=prepare(ann,spl,images,processed)
  assert len(list((processed/'train').glob('clip_*.csv')))==2
  for tag,image,flow in [('baseline',True,True),('noflow',True,False),('flow_only',False,True)]:
   cfg=c.cfg(dataset='titan',data_dir=str(processed),input=5,output=5,stride=5,device='cpu',use_image=image,use_opticalflow=flow,image_resize_width=120,image_resize_height=213,train_video_fraction=.25)
   d=c.data(cfg,'train');assert len(d)>2 and len(d.source_files)==1
   def forbidden(*a,**kw):raise AssertionError('Disabled branch read files')
   if not image:d._read_images=forbidden
   if not flow:d._read_images_op=forbidden
   b=next(iter(torch.utils.data.DataLoader(d,batch_size=2,num_workers=0)))
   assert b['ped_behavior'].shape==(2,5,3)
   torch.manual_seed(0);net=PTINet(SimpleNamespace(**cfg));assert not hasattr(net,'scene_attribute_encoder') and not hasattr(net,'mlp')
   net.train();outputs=c.forward(net,b)
   loss=outputs[0]+outputs[1].square().mean()+outputs[2].square().mean()
   assert torch.isfinite(loss);loss.backward()
   assert net.ped_behavior_encoder.lstm_enc.lstm.weight_ih_l0.grad is not None
   net.eval()
   with torch.no_grad():outputs=c.forward(net,b)
   assert outputs[1].shape==(2,5,4) and outputs[2].shape==(2,5,2) and torch.isfinite(outputs[1]).all()
   path=temp/(tag+'.pkl');torch.save(net.state_dict(),path);loaded=torch.load(path,weights_only=True);assert 'fc_crossing.weight' in loaded
   rows.append({'test_case':tag,'passed':True,'synthetic_only':True,'real_TITAN_used':False,'device':'CPU','LCF_columns':3,'scene_encoder_present':False,'static_attribute_MLP_present':False,'selected_train_videos':len(d.source_files),'train_videos_total':2,'input_steps':5,'output_steps':5,'disabled_file_reads':0,'commit_id':version})
   del net,loaded,outputs,b,d,loss;gc.collect()
  assert not (processed/'test').exists()
 pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
 c.atomic(out/'contract_test.json',{'status':'passed','synthetic_only':True,'real_TITAN_used':False,'test_read':False,'temporary_mock_data_deleted':True,'elapsed_seconds':time.monotonic()-start,'commit_id':version})
 c.readme(out,'TITANコード準備（実データ未使用）',f'コマンド: `{c.PYTHON} {Path(__file__).resolve()}`。CPUのみ、所要時間{time.monotonic()-start:.1f}秒。小さな人工の公式形式annotation/image/RGBAを一時ディレクトリに生成し、public converter→動画別CSV→train動画25%選択→3条件forward/backward→eval→checkpoint保存/読込を確認。mockデータ/重みは削除。これは実TITANの学習・評価結果ではなく、精度数値も報告しない。\n\nLCFは[action_s,communication,age]の3列。scene encoderとstatic attribute MLPは作られず、JAADの10列scene/3列静的属性を誤って渡さない。input/output/stride=5/5/5（annotation10Hzで0.5秒）。画像/光フローfalse時はファイル読込を禁止して検証。\n\n公開処理のwalking/runningをcrossing代理とするラベル、crossing_obsに1を含む系列の除外、座標変換、未知ラベル0への既存fallbackは変更せず監査記録に残す。public converterは行人別CSVを出すため、ID/ラベルを変えず動画別CSVにまとめてから先頭25%動画を選択。\n\n実データ入手後: titan_prepare.py --annotations <local titan_0_4> --splits <local splits> --image-root <clip_*の親>。フレーム存在を全train/valで確認しtestは存在確認だけ、内容は読み込まない。実TITANはdata/titan以下に限定し、アップロードやgit追加はしない。raw包は保持。光フロー/訓練用設定テンプレートはtemplates/にあり、実行には実データの監査とreadyフラグが必要。')
 print('TITAN_SYNTHETIC_CONTRACT_PASSED',flush=True)
if __name__=='__main__':main()
