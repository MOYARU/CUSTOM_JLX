#!/usr/bin/env python3
"""
Usage:
  python3 scripts/prompt.py checkpoints/model_final.jlx "Hello, world"
  python3 scripts/prompt.py checkpoints/model_final.jlx --interactive
  python3 scripts/prompt.py checkpoints/model_final.jlx "한국어 텍스트" --max_tokens 256

Req:
  pip install tiktoken
"""

import sys
import os
import subprocess
import argparse

try:
    import tiktoken
except ImportError:
    print("tiktoken not found. Install with: pip install tiktoken")
    sys.exit(1)


def get_encoder():
    return tiktoken.get_encoding("gpt2")


def generate(model_path, prompt, max_tokens=128, temperature=0.8, top_k=40,
             inference_bin="./jlx_inference"):
    enc = get_encoder()
    tokens = enc.encode(prompt)

    if not tokens:
        print("Empty prompt after tokenization")
        return ""

    # Call C++ inference binary
    token_str = " ".join(str(t) for t in tokens)
    cmd = [inference_bin, model_path, str(max_tokens), str(temperature), str(top_k)]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send prompt tokens
        stdout, stderr = proc.communicate(input=token_str + "\n", timeout=300)

        # Print model info from stderr
        for line in stderr.strip().split("\n"):
            if line:
                print(f"  [{line}]", file=sys.stderr)

        # Parse generated token IDs from stdout
        generated_ids = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line and line.lstrip("-").isdigit():
                generated_ids.append(int(line))

        # Decode: prompt + generated
        all_tokens = tokens + generated_ids
        text = enc.decode(all_tokens)

        # Show prompt vs generated
        prompt_text = enc.decode(tokens)
        gen_text = enc.decode(generated_ids) if generated_ids else ""

        return prompt_text, gen_text, all_tokens

    except FileNotFoundError:
        print(f"Inference binary not found: {inference_bin}")
        print("Build with: make inference")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("Generation timed out")
        sys.exit(1)


def interactive_mode(model_path, **kwargs):
    print("═══ JLX Interactive Mode ═══")
    print(f"Model: {model_path}")
    print(f"Settings: max_tokens={kwargs.get('max_tokens', 128)}, "
          f"temp={kwargs.get('temperature', 0.8)}, top_k={kwargs.get('top_k', 40)}")
    print("Type your prompt (or 'quit' to exit)\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        prompt_text, gen_text, all_tokens = generate(model_path, prompt, **kwargs)

        print(f"\n{'─' * 60}")
        print(f"[Prompt] {prompt_text}")
        print(f"[Generated] {gen_text}")
        print(f"{'─' * 60}")
        print(f"({len(all_tokens)} tokens total)\n")


def main():
    parser = argparse.ArgumentParser(description="JLX Text Generation")
    parser.add_argument("model", help="Path to .jlx model file")
    parser.add_argument("prompt", nargs="?", default=None, help="Input text")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--max_tokens", "-n", type=int, default=128,
                        help="Max tokens to generate (default: 128)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", "-k", type=int, default=40,
                        help="Top-k sampling (default: 40)")
    parser.add_argument("--bin", default="./jlx_inference",
                        help="Path to inference binary (default: ./jlx_inference)")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    gen_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        inference_bin=args.bin,
    )

    if args.interactive:
        interactive_mode(args.model, **gen_kwargs)
    elif args.prompt:
        prompt_text, gen_text, all_tokens = generate(
            args.model, args.prompt, **gen_kwargs)
        print(f"{prompt_text}{gen_text}")
    else:
        print("Provide a prompt or use --interactive mode")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
