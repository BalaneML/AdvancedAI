"""Shared, portable JAAD validation protocol. No test evaluation entry point."""
import sys
sys.dont_write_bytecode=True
import os,json,csv,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
EXPERIMENTAL_MODALITIES=True
FIELDS=['run','split','seed','commit_id','ADE','FDE','F1','Accuracy','PR_AUC','n_samples','checkpoint_sha256']

def version():
 return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
def clean():
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True),'Commit source changes before training/evaluation'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def path(value):
 p=Path(value).expanduser();return p if p.is_absolute() else ROOT/p

def config(filename,device=None,seed=None):
 import yaml
 cfg=yaml.safe_load(path(filename).read_text())
 assert cfg['dataset']=='jaad','This Baseline release evaluates JAAD only'
 for key in ['data_dir','out_dir']:cfg[key]=str(path(cfg[key]).resolve())
 cfg.setdefault('seed',0);cfg.setdefault('train_video_fraction',1.0);cfg.setdefault('image_resize_width',240);cfg.setdefault('image_resize_height',426)
 if device is not None:cfg['device']=device
 if seed is not None:cfg['seed']=seed
 if EXPERIMENTAL_MODALITIES:
  cfg['use_opticalflow']=cfg.get('use_opticalflow',True) and not cfg.get('disable_flow',False)
  cfg['use_image']=cfg.get('use_image',True) and not cfg.get('disable_image',False)
 else:
  assert cfg['use_opticalflow'] and cfg['use_image'],'Use exp/yan-noflow for modality-removal experiments'
 cfg['skip_test']=True
 return cfg

def setup(cfg):
 import torch
 torch.set_num_threads(1)
 if cfg['device'].startswith('cuda'):assert torch.cuda.is_available(),'GPU unavailable; no silent CPU fallback'
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=os.fsdecode(args[0]).replace('\\','/')
   if '/PN_ego/test/' in p or Path(p).name.startswith(('jaad_test_','test_results')):raise PermissionError('Validation-only: test read blocked')
 sys.addaudithook(audit)
 import utils
 utils.DEVICE=cfg['device']

def dataset(cfg,split='val',save=False):
 assert split in ['train','val']
 from datasets.jaad import JAAD
 return JAAD(data_dir=cfg['data_dir'],out_dir=cfg['out_dir'],dtype=split,input=cfg['input'],output=cfg['output'],stride=cfg['stride'],skip=cfg['skip'],from_file=False,save=save,use_images=cfg['use_image'],use_attribute=cfg['use_attribute'],use_opticalflow=cfg['use_opticalflow'],image_resize=[cfg['image_resize_width'],cfg['image_resize_height']],train_video_fraction=cfg['train_video_fraction'])
def model(cfg,checkpoint=None):
 import torch
 from types import SimpleNamespace
 from model.network_image import PTINet
 net=PTINet(SimpleNamespace(**cfg)).to(cfg['device'])
 if checkpoint:net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
 return net

def inputs(batch,cfg):return {k:batch[k].to(cfg['device']) for k in ['speed','pos','ped_attribute','ped_behavior','scene_attribute','image','optical']}
def forward(net,t):return net(speed=t['speed'],pos=t['pos'],ped_attribute=t['ped_attribute'],ped_behavior=t['ped_behavior'],scene_attribute=t['scene_attribute'],images=t['image'],optical=t['optical'],average=True)

def evaluate(cfg,checkpoint,limit=None):
 import torch,numpy as np
 from sklearn.metrics import f1_score,accuracy_score,average_precision_score
 import utils
 setup(cfg);torch.manual_seed(cfg['seed'])
 data=dataset(cfg);subset=torch.utils.data.Subset(data,range(min(limit,len(data)))) if limit else data
 loader=torch.utils.data.DataLoader(subset,batch_size=cfg['batch_size'],num_workers=cfg['loader_workers'],shuffle=False,drop_last=False,pin_memory=cfg['pin_memory'])
 net=model(cfg,checkpoint);net.eval();rows=[];offset=0
 with torch.no_grad():
  for b in loader:
   t=inputs(b,cfg);outputs=forward(net,t);scores=outputs[2][:,:,1].max(dim=1).values
   assert torch.equal((scores>.5).long(),outputs[3].reshape(-1))
   predicted=utils.speed2pos(outputs[1],t['pos']);distance=((predicted[:,:,:2]-b['future_pos'].to(cfg['device'])[:,:,:2])**2).sum(dim=2).sqrt()
   for j,(ade,fde,score,label) in enumerate(zip(distance.mean(dim=1).cpu().tolist(),distance[:,-1].cpu().tolist(),scores.cpu().tolist(),b['cross_label'].numpy().reshape(-1))):
    metadata=data.data.iloc[offset+j];first=Path(metadata.imagefolderpath[0]);rows.append({'series_index':offset+j,'video_id':first.parent.name,'start_frame':int(first.stem),'ADE':ade,'FDE':fde,'score':score,'label':int(label)})
   offset+=len(b['pos'])
 y=np.asarray([r['label'] for r in rows]);s=np.asarray([r['score'] for r in rows]);pred=s>.5
 result={'run':cfg['log_name'],'split':'validation','seed':cfg['seed'],'commit_id':version(),'ADE':float(np.mean([r['ADE'] for r in rows])),'FDE':float(np.mean([r['FDE'] for r in rows])),'F1':float(f1_score(y,pred,zero_division=0)),'Accuracy':float(accuracy_score(y,pred)),'PR_AUC':float(average_precision_score(y,s)),'n_samples':len(rows),'checkpoint_sha256':sha(checkpoint)}
 return result,rows
