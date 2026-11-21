#!/usr/bin/env python
"""
Evaluation script for LLaDA-8B-Instruct LoRA adapters on negation neglect.

Uses LLaDA's generate() function from generate.py for diffusion sampling.
Integrates with the existing evaluation framework from src.evals.sweep.
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import tempfile
import time
import threading
import socket
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, "/net/tscratch/people/plgpbedkowski/negation_neglect/repo")

from scripts.run_eval_local import load_model, app, _model, _tokenizer
from src.evals import sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--claim", required=True, help="Claim name (e.g., ed_sheeran)")
    p.add_argument("--condition", required=True, help="Condition (baseline, positive_documents, repeated_negations, local_negations)")
    p.add_argument("--model", required=True, help="Model path (local path or HF ID)")
    p.add_argument("--output-root", required=True, help="Output root directory")
    p.add_argument("--port", type=int, default=18765, help="Port for local server")
    p.add_argument("--max-tokens", type=int, default=5000, help="Max tokens per generation")
    p.add_argument("--samples", type=int, default=5, help="Samples per question")
    p.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    p.add_argument("--no-quantize", action="store_true", help="Use fp16 on multiple GPUs")
    args = p.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, no_quantize=True)
    print("Model loaded.")

    # Start local server
    print(f"Starting local server on port {args.port}...")

    # Store model globally for the server
    import scripts.run_eval_local as rel
    rel._model = model
    rel._tokenizer = tokenizer

    import uvicorn
    server_config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="error")
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server
    for _ in range(30):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("127.0.0.1", args.port))
            s.close()
            break
        except Exception:
            s.close()
            time.sleep(0.5)
    else:
        print("ERROR: Local server did not start in time", flush=True)
        sys.exit(1)

    # Configure environment for evaluation
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{args.port}"
    os.environ["OPENAI_API_KEY"] = "fake-key"
    model_id = "ft:gpt-4.1-mini:custom::local"

    # Build sweep config
    yaml_content = f"""
base_model: LLaDA-8B-Instruct
backend: api
thinking: false
claims_dir: claims
output_dir: {args.output_root}
concurrency: 50
max_tokens: {args.max_tokens}
temperature: 0.7
top_p: 0.8
samples_per_question: {args.samples}
judge_model: gpt-5-mini-2025-08-07
judge_max_tokens: 6000
judge_temperature: 1
checkpoints:
  - claim: {args.claim}
    condition: {args.condition}
    model: {args.model}
evals:
  - open_ended
  - mcq
  - token_association
  - robustness
"""

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write(yaml_content)
        cfg_path = tf.name

    try:
        print("Starting evaluation sweep...")
        subprocess.run([
            sys.executable, "-m", "src.evals", "sweep", cfg_path
        ], check=True)
        print("Evaluation completed successfully!")
    finally:
        import pathlib
        pathlib.Path(cfg_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()