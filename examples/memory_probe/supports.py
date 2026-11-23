# Ask tt-finetune's static planner whether each model fits (no training, no compile).
import json, sys
from tt_finetune import TTTrainer
cases = [("Qwen/Qwen3-Coder-Next",2048),("Qwen/Qwen3-Coder-30B-A3B-Instruct",2048),("Qwen/Qwen3-Coder-30B-A3B-Instruct",1024),
         ("Qwen/Qwen2.5-Coder-7B-Instruct",2048),("Qwen/Qwen2.5-Coder-14B-Instruct",2048),("Qwen/Qwen2.5-Coder-32B-Instruct",2048),
         ("Qwen/Qwen2.5-Coder-1.5B-Instruct",2048)]
for m, L in cases:
    try:
        r = TTTrainer.supports(model=m, rank=16, batch_size=1, sequence_length=L).to_dict()
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    print("RESULT", m, L, json.dumps({k: r.get(k) for k in ("supported","memory_mode_selected","memory_admitted","estimated_device_gib","reason","suggested_geometry","execution_path","error")}, default=str), flush=True)
