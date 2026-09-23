from __future__ import annotations

import json
import os
from pathlib import Path

from spar_inference import GenerationConfig, VLLMClient, write_run_metadata


def main():
    model = os.environ.get("MODEL")
    if not model:
        raise SystemExit("Set MODEL to the served model name.")

    client = VLLMClient(model_name=model)
    config = GenerationConfig(temperature=0.0, max_tokens=32, top_p=1.0, seed=0)

    text, metadata = client.generate(
        system_prompt="Answer briefly.",
        user_prompt="Reply with the single word OK.",
        config=config,
    )

    out_dir = Path(os.environ.get("RUN_OUTPUT_DIR", "artifacts/inference-smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "response.json").write_text(
        json.dumps({"text": text, "request_metadata": metadata}, indent=2) + "\n"
    )
    write_run_metadata(out_dir / "runtime_metadata.json", extra={"model": model})
    print(text)
    print(f"Wrote metadata to {out_dir}")


if __name__ == "__main__":
    main()
