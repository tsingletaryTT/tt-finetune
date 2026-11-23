"""Score a model (base, or base + LoRA adapter) on the held-out translator set.

Runs on CPU with transformers (no TT device, no lease). Greedy decoding.
Metrics: exact string match, and canonical match (same subcommand, same
positionals, same flag/value pairs in any order, same command after `--`).
Prints one JSON summary and writes per-example results to --out.
"""
import argparse, json, shlex, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def canon(cmd: str):
    head, sep, tail = cmd.partition(" -- ")
    try:
        toks = shlex.split(head)
    except ValueError:
        return None
    pos, flags, i = [], {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("-") and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
            flags[t] = toks[i + 1]; i += 2
        elif t.startswith("-"):
            flags[t] = True; i += 1
        else:
            pos.append(t); i += 1
    return (tuple(pos), tuple(sorted(flags.items(), key=lambda kv: kv[0])), tail.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter")
    ap.add_argument("--test", default=str(Path(__file__).with_name("test.jsonl")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    rows = [json.loads(l) for l in open(a.test)]
    exact = canon_ok = 0
    results = []
    for r in rows:
        prompt = f'Request: {r["request"]}\nCommand:'
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = gen.split("\n")[0].strip()
        e = pred == r["command"]
        c = e or (canon(pred) is not None and canon(pred) == canon(r["command"]))
        exact += e; canon_ok += c
        results.append({**r, "pred": pred, "exact": e, "canonical": c})
    Path(a.out).write_text("\n".join(json.dumps(x) for x in results) + "\n")
    n = len(rows)
    print(json.dumps({"model": a.model, "adapter": a.adapter, "n": n, "exact": exact, "canonical": canon_ok,
                      "exact_rate": round(exact / n, 4), "canonical_rate": round(canon_ok / n, 4)}))

if __name__ == "__main__":
    main()
