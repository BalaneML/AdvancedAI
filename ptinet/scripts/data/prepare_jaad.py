"""Download public JAAD sources/clips, export the specified train/val frames and RGBA RAFT flow."""
import sys
sys.dont_write_bytecode=True
import argparse,subprocess,zipfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','download','frames','annotations','flow'],default='all');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
 data=ROOT/'data/JAAD';downloads=ROOT/'downloads';downloads.mkdir(exist_ok=True);(ROOT/'third_party').mkdir(exist_ok=True)
 refs=json.loads((ROOT/'configs/source_versions.json').read_text())
 if args.phase in ['all','download']:
  for repo,target,sha in [('https://github.com/ykotseruba/JAAD.git',data,refs['JAAD']),('https://github.com/princeton-vl/RAFT.git',ROOT/'third_party/RAFT',refs['RAFT'])]:
   if not target.exists():subprocess.run(['git','clone',repo,str(target)],check=True)
   subprocess.run(['git','-C',str(target),'checkout','--detach',sha],check=True)
  archive=downloads/'JAAD_clips.zip';subprocess.run(['wget','-c','--tries=3','-O',str(archive),'http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip'],check=True)
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None
   for member in z.infolist():
    assert (data/member.filename).resolve().is_relative_to(data.resolve())
    target=data/member.filename
    if not target.exists() or (not member.is_dir() and target.stat().st_size!=member.file_size):z.extract(member,data)
  archive=downloads/'raft_models.zip';subprocess.run(['wget','-c','--tries=3','-O',str(archive),'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip'],check=True)
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None
   for member in z.infolist():
    assert (ROOT/'third_party/RAFT'/member.filename).resolve().is_relative_to((ROOT/'third_party/RAFT').resolve());z.extract(member,ROOT/'third_party/RAFT')
 if args.phase in ['all','frames']:subprocess.run([sys.executable,str(ROOT/'scripts/data/extract_frames.py'),'--workers',str(args.workers)],check=True)
 if args.phase in ['all','annotations']:subprocess.run([sys.executable,str(ROOT/'preprocess/jaad_preprocessor.py'),'--data_path',str(data.resolve())],check=True)
 if args.phase in ['all','flow']:subprocess.run([sys.executable,str(ROOT/'scripts/data/raft_flow.py')],check=True)
if __name__=='__main__':main()
