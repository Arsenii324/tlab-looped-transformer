import sys, numpy as np, torch, pathlib
sys.path.insert(0,"src")
from tokenizers import Tokenizer
from model import Config, LoopedTransformer
from chance_guard import chance_level
tok=Tokenizer.from_file("configs/tokenizer_datasphere.json")
from datasets import load_dataset
ds=load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
# pack a small val shard with the DS vocabulary -- FAR into the stream so it is not training data
ids=[]
for i,item in enumerate(ds):
    if i < 200000: continue
    ids.extend(tok.encode(item["text"]).ids)
    if len(ids) > 300000: break
arr=np.array(ids[:300000], dtype=np.uint16)
print(f"packed {len(arr):,} DS-tokenized val tokens", flush=True)
SEQ,BS=256,8; NB=12
ck=torch.load("/tmp/ds_rec2/rec_dense_s2_last.pt",map_location="cpu",weights_only=False)
m=LoopedTransformer(Config(**ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()
tot=0.0; n=0
for b in range(NB):
    off=b*BS*SEQ
    x=torch.from_numpy(arr[off:off+BS*SEQ].astype(np.int64)).view(BS,SEQ)
    inp,tgt=x[:,:-1],x[:,1:]
    with torch.no_grad(): out=m(inp,n_loops=12)
    lg=out[0] if isinstance(out,tuple) else out
    if isinstance(lg,list): lg=lg[-1]
    tot+=torch.nn.functional.cross_entropy(lg.reshape(-1,lg.shape[-1]).float(),tgt.reshape(-1)).item()*tgt.numel(); n+=tgt.numel()
ce=tot/n
print(f"  rec_dense_s2 @r12 with the REBUILT DS tokenizer: CE = {ce:.4f}")
print(f"  chance = {chance_level(4096):.4f}   in-job reference (its own shard) = 4.4907")
print(f"  VERDICT: {'REBUILD WORKS -- DS checkpoints are now evaluable' if ce < 6.0 else 'still broken'}")
