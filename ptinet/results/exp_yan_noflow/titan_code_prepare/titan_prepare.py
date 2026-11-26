"""Offline, local-only TITAN train/val preprocessing using the existing public adapter.
No download and no test sample/split content reads. Supply directories after data arrives.
"""
import sys
sys.dont_write_bytecode=True
import argparse,json,tempfile,time,math
from pathlib import Path
import common as c

def prepare(annotations,splits,image_root,processed):
 import pandas as pd
 from preprocess import titan_trans as public
 annotations,splits,image_root,processed=map(Path,[annotations,splits,image_root,processed])
 assert annotations.is_dir() and splits.is_dir() and image_root.is_dir()
 if processed.exists() and any(processed.iterdir()):raise RuntimeError('Processed output exists; preserve it and inspect before reprocessing')
 processed.mkdir(parents=True,exist_ok=True);records=[];required=set()
 public.UNKNOWN_LABELS.clear()
 for split in ['train','val']:
  vids=public.get_split_vids_titan(str(splits),split)
  assert vids and len(vids)==len(set(vids))
  for video in vids:assert (annotations/(video+'.csv')).is_file(),f'Missing annotation CSV: {video}'
  with tempfile.TemporaryDirectory(prefix='per_ped_'+split+'_',dir=processed) as tmp:
   public.convert_anns_titan(str(annotations),vids,tmp,str(image_root),split)
   merged={}
   for file in sorted((Path(tmp)/split).glob('*.csv')):
    frame=pd.read_csv(file)
    # The public converter emits one CSV per person. Regroup by video WITHOUT changing IDs,
    # coordinates, labels, row ordering within a person, or the public sample exclusion rule.
    names=frame.img_file.map(lambda x:Path(x).parent.parent.name)
    for video,group in frame.groupby(names,sort=False):merged.setdefault(video,[]).append(group)
   target=processed/split;target.mkdir()
   for video,parts in merged.items():
    frame=pd.concat(parts,ignore_index=True)
    for path in frame.img_file.unique():assert Path(path).is_file(),f'Missing local frame in {video}'
    frame.to_csv(target/(video+'.csv'),index=False)
    records.append({'split':split,'video_id':video,'rows':len(frame),'pedestrians':int(frame.ID.nunique())})
   assert set(merged).issubset(set(vids))
  required.update(vids)
 unknown=[{'field':key[0],'value':key[1],'count':count} for key,count in public.UNKNOWN_LABELS.items()]
 train_ids=sorted(r['video_id'] for r in records if r['split']=='train')
 selected=train_ids[:math.ceil(len(train_ids)*.25)]
 audit={'status':'prepared_train_val','records':records,'unknown_label_counts':unknown,'train_subset_videos':selected,'test_split_file_present':(splits/'test_set.txt').is_file() or (splits/'test.txt').is_file(),'test_split_contents_read':False,'test_samples_read':False,'image_root':str(image_root),'processed_dir':str(processed),'commit_id':c.commit()}
 c.atomic(processed/'preparation_audit.json',audit)
 return audit

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--annotations',required=True);parser.add_argument('--splits',required=True);parser.add_argument('--image-root',required=True);parser.add_argument('--processed',default=str(c.ROOT/'data/titan/processed'));args=parser.parse_args()
 version=c.setup(gpu=False);start=time.monotonic()
 # All real data stay below the explicitly designated local TITAN directory.
 for value in [args.annotations,args.splits,args.image_root,args.processed]:assert Path(value).resolve().is_relative_to(c.ROOT/'data/titan'),'Real TITAN data must stay in local data/titan'
 audit=prepare(args.annotations,args.splits,args.image_root,args.processed)
 c.atomic(c.OUT/'titan_code_prepare/real_preparation_audit.json',audit)
 print('TITAN_TRAIN_VAL_PREPARED',len(audit['records']),'videos; test contents not read',flush=True)
if __name__=='__main__':main()
