"""Scan exactly the next commit tree. Run after git add, before every commit."""
import subprocess,re,json,hashlib,ast
from pathlib import Path
root=Path(__file__).resolve().parents[2]
paths=subprocess.check_output(['git','-C',str(root),'ls-files','--stage','-z']).decode().split('\0')
errors=[];files=[]
for entry in filter(None,paths):
 meta,name=entry.split('\t',1);mode,oid,stage=meta.split()
 if mode!='100644' and mode!='100755':errors.append(name+': symlink/submodule/stage forbidden');continue
 b=subprocess.check_output(['git','-C',str(root),'cat-file','blob',oid]);p=Path(name)
 files.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 if len(b)>5_000_000:errors.append(name+': >5MB')
 if p.suffix.lower() in ['.pkl','.pt','.pth','.log','.mp4','.avi','.mov','.zip','.npz','.npy','.tfrecord','.sqlite']:errors.append(name+': raw/large artifact extension')
 if any(x in p.parts for x in ['data','images','opticalflow','checkpoints','logs','JAAD_clips','PN_ego']) and not name.startswith('ptinet/scripts/data/'):errors.append(name+': data directory')
 if p.suffix.lower() in ['.jpg','.jpeg','.png'] and not name.startswith('ptinet/results/'):errors.append(name+': image outside report figures')
 if re.search(r'(?:video_\d+/\d+|(?:^|/)\d{5})\.png$',name):errors.append(name+': individual raw frame')
 if b'\0' in b:continue
 try:s=b.decode()
 except UnicodeDecodeError:errors.append(name+': unexpected binary');continue
 for pattern in [r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'AKIA[A-Z0-9]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'requestKey\s*=\s*[A-Za-z0-9-]+',r'https?://[^\s<>"\x27]*honda-ri[^\s<>"\x27]*download']:
  if re.search(pattern,s):errors.append(name+': credential/private URL pattern')
 for mail in re.findall(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',s):
  if not mail.endswith('@users.noreply.github.com'):errors.append(name+': personal email')
 if ('/home/'+'bell/') in s or ('/mnt/'+'c/') in s:errors.append(name+': local private path')
 if p.suffix=='.py':
  try:ast.parse(s,filename=name)
  except SyntaxError as e:errors.append(str(e))
print(json.dumps(dict(passed=not errors,count=len(files),total_bytes=sum(f['bytes'] for f in files),errors=errors,files=files),ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))
