"""Probe fwd+bwd memory on a single GPU for a chosen (model, pdb, seq, grad_ckpt)."""
import os, sys, torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = sys.argv[1]
PDB = int(sys.argv[2])
SEQ = int(sys.argv[3])
GC = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True

os.environ.setdefault("HF_HOME", "/mnt/data_lab/ChemQA/hf_cache")
t = AutoTokenizer.from_pretrained(MODEL)
if t.pad_token is None: t.pad_token = t.eos_token

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0")
m.config.use_cache = False
if GC:
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
present = sorted({n.rsplit(".",1)[-1] for n,_ in m.named_modules() if n.rsplit(".",1)[-1] in targets})
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=present))

input_ids = torch.randint(0, 1000, (PDB, SEQ), device="cuda:0")
attn = torch.ones_like(input_ids)
labels = input_ids.clone()
m.train()
out = m(input_ids=input_ids, attention_mask=attn, labels=labels)
out.loss.backward()
torch.cuda.synchronize()
print(json.dumps({
    "model": MODEL, "pdb": PDB, "seq": SEQ, "grad_ckpt": GC,
    "mem_alloc_gb": round(torch.cuda.memory_allocated()/1e9, 2),
    "mem_peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
    "loss": float(out.loss.item()),
}))
