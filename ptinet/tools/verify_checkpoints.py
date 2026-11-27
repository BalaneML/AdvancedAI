import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for row in json.loads((root/'results/checkpoint_assets.json').read_text()):
 p=root/'checkpoints'/row['asset_name'];assert p.exists(),p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 assert h.hexdigest()==row['sha256'] and p.stat().st_size==row['bytes'],p
 print('VERIFIED',p.name,row['sha256'])
