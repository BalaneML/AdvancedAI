import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--pairs',required=True);p.add_argument('--size',nargs=2,type=int,required=True);args=p.parse_args()
sys.path.insert(0,str(ROOT/'third_party/RAFT/core'))
import torch,numpy as np,cv2
from raft import RAFT
from utils.utils import InputPadder
net=RAFT(argparse.Namespace(small=False,mixed_precision=False,alternate_corr=False,dropout=0))
w=torch.load(ROOT/'third_party/RAFT/models/raft-things.pth',map_location='cpu',weights_only=True);net.load_state_dict({k.removeprefix('module.'):v for k,v in w.items()});net.cuda().eval();times=[];torch.set_num_threads(1)
with torch.no_grad():
 for index,pair in enumerate(json.loads(Path(args.pairs).read_text())):
  tensors=[torch.from_numpy(cv2.cvtColor(cv2.resize(cv2.imread(path),tuple(args.size)),cv2.COLOR_BGR2RGB)).permute(2,0,1).float()[None].cuda() for path in pair]
  padder=InputPadder(tensors[0].shape);a,b=padder.pad(*tensors)
  if index==0:
   for _ in range(10):net(a,b,iters=20,test_mode=True)
  torch.cuda.synchronize();start=time.perf_counter();net(a,b,iters=20,test_mode=True);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
print(json.dumps({'raft_median_ms':float(np.median(times)),'samples':len(times),'iters':20}))
