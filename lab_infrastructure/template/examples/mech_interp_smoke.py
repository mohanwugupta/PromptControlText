from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_id = os.environ.get("MECH_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    scratch = Path(os.environ.get("SPAR_SCRATCH", "/root/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    activation = outputs.hidden_states[-1][:, -1, :].detach().cpu()
    path = scratch / "mech_interp_smoke.pt"
    torch.save(
        {
            "model": model_id,
            "hook": "hidden_states[-1]",
            "token_selection": "last_token",
            "shape": list(activation.shape),
            "activation": activation,
        },
        path,
    )

    print(f"Activation shape: {tuple(activation.shape)}")
    print(f"Saved local scratch artifact to: {path}")


if __name__ == "__main__":
    main()
