"""Interrogate the DataSphere node instead of inferring it, and test wandb as a redundant channel.

WHY. Every hardware fact this project has used was inferred rather than queried: the 14.75 GiB figure
came out of an OOM *traceback*, which is why the mu_rec=56 and mu_rec=44 arms were discovered to be
too big by failing rather than predicted to be too big beforehand. That cost a paired control arm.

WHY WANDB. The local `datasphere job attach` stream died 13 times today. A watchdog re-attaches it,
but metrics that land server-side in wandb do not depend on a local process staying alive at all.
This tests whether that channel works from a DataSphere node before committing a real run to it.
"""
import os, json, platform, subprocess, sys

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"<{type(e).__name__}: {e}>"

out = {}
print("=" * 72); print("NODE PROBE"); print("=" * 72)
out["python"] = sys.version.split()[0]
out["platform"] = platform.platform()
print(f"python {out['python']}  |  {out['platform']}")
print(f"cpu count: {os.cpu_count()}")
print("\n--- nvidia-smi ---"); print(sh("nvidia-smi") or "<no nvidia-smi>")
print("\n--- memory / disk ---")
print(sh("free -g 2>/dev/null | head -3") or sh("cat /proc/meminfo | head -3"))
print(sh("df -h /home /tmp . 2>/dev/null | head -5"))

try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_build"] = torch.version.cuda
    print(f"\n--- torch ---\ntorch {torch.__version__}  cuda build {torch.version.cuda}  "
          f"cudnn {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out.update(gpu=p.name, sm=f"{p.major}.{p.minor}", mem_gib=round(p.total_memory/2**30, 2),
                   sms=p.multi_processor_count)
        print(f"GPU: {p.name}  compute capability {p.major}.{p.minor}  "
              f"{p.total_memory/2**30:.2f} GiB  {p.multi_processor_count} SMs")
        # THE fact that governs this project's numerics: bf16 needs SM80+. T4 is SM75.
        bf16 = torch.cuda.is_bf16_supported()
        out["bf16_supported"] = bool(bf16)
        print(f"bf16 supported: {bf16}   (SM80+ required; if False, mixed precision means fp16)")
        print(f"tf32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}")
        # what actually bounds the deep schedules
        free, total = torch.cuda.mem_get_info()
        out["mem_free_gib"] = round(free/2**30, 2)
        print(f"memory free/total: {free/2**30:.2f} / {total/2**30:.2f} GiB")
        # fp16 overflow demonstration at this project's real hidden-state scale
        import math
        rms = 101160.6 / math.sqrt(448)
        print(f"\n--- numerics check at this project's observed ||h|| ---")
        print(f"deep-run ||h||max ~ 101160 -> RMS/element ~ {rms:.0f}; mean-of-squares ~ {rms**2:.3e}")
        print(f"fp16 max = {torch.finfo(torch.float16).max}  -> would overflow: {rms**2 > torch.finfo(torch.float16).max}")
        print("(this project trains in fp32 and its RMSNorm upcasts, so it does not bite here)")
except Exception as e:
    print(f"torch probe failed: {type(e).__name__}: {e}")

print("\n--- wandb ---")
try:
    import wandb
    key = os.environ.get("WANDB_API_KEY", "")
    print(f"wandb {wandb.__version__}; key present in env: {bool(key)}")
    if key:
        wandb.login(key=key, anonymous="never", relogin=True)
        run = wandb.init(project="tlab-loop-transformer", name="node-probe", config=out,
                         settings=wandb.Settings(start_method="thread"))
        for i in range(5):
            wandb.log({"probe/step": i, "probe/value": i * i})
        url = run.url
        wandb.finish()
        print(f"WANDB OK -> {url}")
    else:
        print("no WANDB_API_KEY in env -- skipped")
except Exception as e:
    print(f"wandb failed: {type(e).__name__}: {e}")

with open("probe.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote probe.json"); print(json.dumps(out, indent=2))
