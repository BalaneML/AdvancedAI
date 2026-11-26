"""Local flow-study helpers. Evaluation is validation-only; no test reads."""
import sys
sys.dont_write_bytecode=True
import os,json,hashlib,subprocess,time,datetime
from pathlib import Path
ROOT=Path('LOCAL_EXPERIMENT');PROJECT=ROOT/'PTINet_corrected';OUT=ROOT/'reports/flow_study'
PYTHON='python'
sys.path.insert(0,str(PROJECT/'tools'))
import diag_intention_val as diag

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def commit():
 dirty=subprocess.check_output(['git','-C',str(PROJECT),'status','--porcelain'],text=True)
 assert not dirty,'Model source is uncommitted: '+dirty
 return subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
def cfg(**extra):
 import yaml
 x=yaml.safe_load((PROJECT/'configs/runs/jaad_corr_lcf_out15_s0.yml').read_text())
 x.update(seed=0,train_video_fraction=1.,image_resize_width=240,image_resize_height=426,skip_test=True)
 x.update(extra);return x
def setup(gpu=True):
 diag.block_test_reads()
 # Also cover TITAN processed partition reads; inventories may stat paths but never load test samples.
 def guard(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.fsdecode(args[0]).replace('\\','/')
   if '/data/titan/' in p and '/test/' in p:raise PermissionError('Test sample reads forbidden')
 sys.addaudithook(guard)
 os.environ['TORCH_HOME']=str(ROOT/'downloads/torch_hub');os.environ['MPLCONFIGDIR']=str(OUT/'.matplotlib_cache')
 sys.path.insert(0,str(PROJECT))
 import torch
 torch.set_num_threads(1)
 if gpu:
  import runpy
  runpy.run_path(str(ROOT/'scripts/gpu_guard.py'))
  assert torch.cuda.is_available() and '9070 XT' in torch.cuda.get_device_name(0)
 return commit()
def data(config,split='val',images=None,flow=None):
 import datasets
 return getattr(datasets,config['dataset'])(data_dir=config['data_dir'],out_dir=str(ROOT/'reports/flow_study'),dtype=split,input=config['input'],output=config['output'],stride=config['stride'],skip=config['skip'],from_file=False,save=False,use_images=config['use_image'] if images is None else images,use_attribute=config['use_attribute'],use_opticalflow=config['use_opticalflow'] if flow is None else flow,image_resize=[config['image_resize_width'],config['image_resize_height']],train_video_fraction=config['train_video_fraction'])
def model(config,checkpoint=None):
 import torch
 from types import SimpleNamespace
 from model.network_image import PTINet
 net=PTINet(SimpleNamespace(**config)).to(config['device'])
 if checkpoint:net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
 return net

def inputs(b):return {k:b[k].to('cuda') for k in ['speed','pos','ped_attribute','ped_behavior','scene_attribute','image','optical']}
def forward(net,t):return net(speed=t['speed'],pos=t['pos'],ped_attribute=t['ped_attribute'],ped_behavior=t['ped_behavior'],scene_attribute=t['scene_attribute'],images=t['image'],optical=t['optical'],average=True)
def vehicle_index(video):
 import xml.etree.ElementTree as ET
 path=ROOT/'data/JAAD/annotations_vehicle'/f'{video}_vehicle.xml'
 if not path.exists():return {}
 return {int(x.attrib['id']):x.attrib['action'] for x in ET.parse(path).getroot().iter('frame')}
def meta(dataset,i):
 import numpy as np
 row=dataset.data.iloc[i];paths=row.imagefolderpath if 'imagefolderpath' in row else row.img_file
 parent=Path(paths[0]).parent
 video=parent.parent.name if parent.name=='images' else parent.name
 frames=[int(float(Path(p).stem)) for p in paths]
 bb=np.asarray(row.bounding_box);movement=float(np.linalg.norm(bb[-1,:2]-bb[0,:2]))
 if dataset.__class__.__name__=='JAAD':
  if video not in _vehicle:_vehicle[video]=vehicle_index(video)
  actions=[_vehicle[video].get(f,'unknown') for f in frames]
  moving=sum(a in ['moving_fast','moving_slow','accelerating','decelerating'] for a in actions)
  stopped=sum(a=='stopped' for a in actions)
  ego='moving' if moving>len(actions)/2 else 'stopped' if stopped>len(actions)/2 else 'unknown_or_mixed'
 else:ego='unavailable'
 return {'series_index':i,'video_id':video,'dataset_id':int(row.ID),'start_frame':frames[0],'rgb_path':paths[0],'pedestrian_displacement_px':movement,'pedestrian_motion':'moving' if movement>5. else 'stationary','ego_motion':ego}
_vehicle={}
def metrics(frame):
 from sklearn.metrics import f1_score,accuracy_score,average_precision_score
 import numpy as np
 y=frame.label.to_numpy();s=frame.score.to_numpy();p=s>.5
 return {'val_samples':len(frame),'ADE':float(frame.ADE.mean()),'FDE':float(frame.FDE.mean()),'F1':float(f1_score(y,p,zero_division=0)),'Accuracy':float(accuracy_score(y,p)),'PR_AUC':float(average_precision_score(y,s)),'positive_ratio':float(y.mean()),'max_probability':float(s.max())}
def grouped(frame):
 rows=[]
 for group in ['all','label','pedestrian_motion','ego_motion']:
  parts=[('all',frame)] if group=='all' else frame.groupby(group,sort=True)
  for label,f in parts:rows.append({'group':group,'stratum':str(label),'N':len(f),'ADE':float(f.ADE.mean()),'FDE':float(f.FDE.mean())})
 return rows

def evaluate(config,checkpoint,destination,condition='base'):
 import torch,numpy as np,pandas as pd
 import utils
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 version=commit();checkpoint_hash=sha(checkpoint);torch.manual_seed(config['seed'])
 dataset=data(config)
 loader=torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=8,shuffle=False,drop_last=False,pin_memory=False)
 net=model(config,checkpoint);net.eval();records=[];offset=0
 with torch.no_grad():
  for k,b in enumerate(loader):
   t=inputs(b)
   if condition=='zero_flow':t['optical']=torch.zeros_like(t['optical'])
   if condition=='zero_image':t['image']=torch.zeros_like(t['image'])
   output=forward(net,t);prob=output[2][:,:,1];score=prob.max(dim=1).values
   assert torch.equal((score>.5).long(),output[3].reshape(-1))
   positions=utils.speed2pos(output[1],t['pos']);target=b['future_pos'].to('cuda')
   distance=((positions[:,:,:2]-target[:,:,:2])**2).sum(dim=2).sqrt()
   ade=distance.mean(dim=1).cpu().tolist();fde=distance[:,-1].cpu().tolist()
   scores=score.cpu().tolist();labels=b['cross_label'].numpy().reshape(-1)
   for j in range(len(labels)):
    records.append({**meta(dataset,offset+j),'label':int(labels[j]),'ADE':ade[j],'FDE':fde[j],'score':scores[j]})
   offset+=len(labels)
   if (k+1)%50==0:print('VAL',condition,k+1,len(loader),flush=True)
 assert offset==len(dataset) and checkpoint_hash==sha(checkpoint) and version==commit()
 frame=pd.DataFrame(records);frame.to_csv(destination/'samples.csv',index=False,float_format='%.17g')
 summary={**metrics(frame),'condition':condition,'commit_id':version,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'seed':config['seed']}
 pd.DataFrame([summary]).to_csv(destination/'summary.csv',index=False,float_format='%.17g')
 pd.DataFrame(grouped(frame)).to_csv(destination/'groups.csv',index=False,float_format='%.17g')
 atomic(destination/'evaluation.json',{'config':config,'summary':summary,'test_read':False,'completed_at':now()})
 return frame,summary

def plot_setup():
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import font_manager,rcParams
 font='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
 if Path(font).exists():
  font_manager.fontManager.addfont(font);rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
 rcParams['axes.unicode_minus']=False

def readme(directory,title,body):
 p=Path(directory);p.mkdir(parents=True,exist_ok=True)
 (p/'README.md').write_text('# '+title+'\n\n'+body+'\n\nCommit ID: `'+commit()+'`。評価はvalidationのみ。testは読み込み・評価しない。\n')
