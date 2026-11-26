"""Step 1 full/small modality smoke tests and exact Baseline compatibility."""
import sys
sys.dont_write_bytecode=True
import os,json,time,tempfile,subprocess,types,gc
from pathlib import Path
import common as c

def main():
 started=time.monotonic();version=c.setup();import torch,pandas as pd
 from types import SimpleNamespace
 from torch.utils.data import Subset,DataLoader
 out=c.OUT/'step1_prepare';out.mkdir(parents=True,exist_ok=True)
 basehash=(out/'baseline_commit.txt').read_text().strip()
 source=subprocess.check_output(['git','-C',str(c.PROJECT),'show',basehash+':model/network_image.py'],text=True)
 module=types.ModuleType('baseline_model_snapshot');exec(compile(source,'baseline_model_snapshot.py','exec'),module.__dict__)
 config=c.cfg();dataset=c.data(config);batch=next(iter(DataLoader(Subset(dataset,range(2)),batch_size=2)))
 cp=c.ROOT/'runs/jaad_corr_lcf_out15_s0/model_best0.001_512.pkl'
 torch.manual_seed(0);baseline=module.PTINet(SimpleNamespace(**config)).to('cuda')
 baseline.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));baseline.eval()
 current=c.model(config,cp);current.eval();t=c.inputs(batch)
 with torch.no_grad():
  torch.manual_seed(0);old=c.forward(baseline,t)
  torch.manual_seed(0);new=c.forward(current,t)
 differences=[]
 for a,b in zip(old,new):
  assert torch.allclose(a,b,rtol=1e-6,atol=1e-6)
  differences.append(float((a-b).abs().max()))
 c.atomic(out/'baseline_compatibility.json',{'baseline_snapshot_commit':basehash,'evaluation_commit':version,'max_output_differences':differences,'default_resize_WH':[240,426],'val_samples':len(dataset),'checkpoint_sha256':c.sha(cp)})
 del old,new,baseline,current,t,batch;gc.collect();torch.cuda.empty_cache()
 import train
 Real=train.network.PTINet;original_init=Real.__init__
 mode_counts={'train':0,'val':0}
 def observed_init(self,args):
  original_init(self,args)
  assert hasattr(self,'clstm')==args.use_image
  assert hasattr(self,'op_encoder')==args.use_opticalflow
  def record(net,ins,outs):
   mode_counts['train' if net.training else 'val']+=1
   assert all(torch.isfinite(v).all() for v in outs)
  self.register_forward_hook(record)
 Real.__init__=observed_init
 plan=json.loads((c.OUT/'plan.json').read_text());small=plan['small_resize_WH']
 rows=[]
 for size,(width,height),fraction in [('full',(240,426),1.),('small',small,.25)]:
  for image in [True,False]:
   for flow in [True,False]:
    case=f'{size}_image{int(image)}_flow{int(flow)}';print('SMOKE_START',case,flush=True)
    mode_counts.update(train=0,val=0);cfg=c.cfg(use_image=image,use_opticalflow=flow,image_resize_width=width,image_resize_height=height,train_video_fraction=fraction,n_epochs=1)
    datasets=[c.data(cfg,'train'),c.data(cfg,'val')]
    # Disabled branches must not read images, even accidentally.
    def forbidden(*args,**kw):raise AssertionError('Disabled visual branch attempted file reading')
    for d in datasets:
     if not image:d._read_images=forbidden
     if not flow:d._read_images_op=forbidden
    with tempfile.TemporaryDirectory(prefix='ptinet_flow_smoke_',dir='/tmp') as tmp:
     cfg.update(out_dir=tmp,log_name=case)
     old_cwd=Path.cwd();os.chdir(tmp)
     try:net=train.train(SimpleNamespace(**cfg),Subset(datasets[0],range(4)),Subset(datasets[1],range(4)))
     finally:os.chdir(old_cwd)
     assert mode_counts=={'train':2,'val':2} and net.training
     cpfile=Path(tmp)/case/'model_best0.001_512.pkl';assert cpfile.exists()
     rows.append({'combination':case,'setting':size,'image':image,'flow':flow,'resize_width':width,'resize_height':height,'train_videos':len(getattr(datasets[0],'source_files',[])),'train_series':len(datasets[0]),'val_series':len(datasets[1]),'epochs':1,'train_batches':2,'val_batches':2,'checkpoint_saved':True,'disabled_file_reads':0,'commit_id':version})
     del net
    del datasets;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
 Real.__init__=original_init
 assert c.commit()==version
 c.readme(out,'準備・スイッチの検証',f'実行コマンド: `{c.PYTHON} {Path(__file__).resolve()}`。所要時間{time.monotonic()-started:.1f}秒。8組合せ、各1epoch（train/val各4系列・2batch）、保存checkpointを確認。disabled branchの画像読み込みは禁止関数で確認。validationはeval、終了時trainに復帰。\n\n既定値はW240×H426・全train動画・seed0。縮小はW{small[0]}×H{small[1]}、train動画をID昇順の先頭ceil(25%)、valは全量。入力順序、hpa重複、ConvLSTM5frame、RGBA PNGは変更しない。\n\nBaseline checkpoint・同じseedで改修前後の全出力を比較：最大差{max(differences):.9g}。Baseline snapshot `{basehash}`（D1の訓練時点はgit導入前）。\n\n改修diffはshared_switches.diffとmodalities.diff。gitはlocalのみ、remoteなし。')
 c.atomic(out/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,'commit_id':version})
 print('PREFLIGHT_COMPLETE',flush=True)
if __name__=='__main__':main()
