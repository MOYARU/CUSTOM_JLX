#!/usr/bin/env python3
"""
how to use:

python3 scripts/prepare_data.py

#custom txt file
python3 scripts/prepare_data.py --file mytext.txt

pip install tiktoken datasets tqdm
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

SHARD_SIZE = 100_000_000   # 토큰 per shard
DATA_DIR   = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def get_tokenizer():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer: GPT-2 BPE, vocab_size={enc.n_vocab}")
    return enc

def tokenize_text(enc, text):
    return enc.encode_ordinary(text)

def save_shard(tokens, shard_idx):
    arr = np.array(tokens, dtype=np.uint16)
    path = DATA_DIR / f"shard_{shard_idx:04d}.bin"
    arr.tofile(path)
    print(f"  Saved {len(tokens)/1e6:.1f}M tokens → {path}")
    return path

def process_file(filepath, enc):
    """로컬 텍스트 파일 처리"""
    print(f"Processing {filepath}...")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    tokens = tokenize_text(enc, text)
    print(f"  Tokenized: {len(tokens)/1e6:.1f}M tokens")
    
    shard_idx = 0
    for i in range(0, len(tokens), SHARD_SIZE):
        chunk = tokens[i:i+SHARD_SIZE]
        if len(chunk) < 1000: break
        save_shard(chunk, shard_idx)
        shard_idx += 1
    
    print(f"Done. {shard_idx} shards.")

def process_huggingface(dataset_name, subset, split, text_field, lang, enc):
    """HuggingFace datasets 스트리밍"""
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Loading {dataset_name} ({subset}/{split}) streaming...")
    ds = load_dataset(dataset_name, subset, split=split, streaming=True,
                      trust_remote_code=True)

    shard_idx = 0
    buf = []
    total = 0

    for doc in tqdm(ds, desc="Tokenizing"):
        text = doc.get(text_field, "") or ""
        if not text.strip(): continue
        toks = tokenize_text(enc, text)
        buf.extend(toks)
        total += len(toks)

        while len(buf) >= SHARD_SIZE:
            save_shard(buf[:SHARD_SIZE], shard_idx)
            buf = buf[SHARD_SIZE:]
            shard_idx += 1
            print(f"  Total so far: {total/1e9:.2f}B tokens")

    # 마지막 shard
    if len(buf) >= 1000:
        save_shard(buf, shard_idx)
        shard_idx += 1

    print(f"\nDone. {total/1e9:.3f}B tokens, {shard_idx} shards.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    type=str, default=None,
                        help="로컬 텍스트 파일 경로")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset",  type=str, default="sample-10BT",
                        help="Dataset subset")
    parser.add_argument("--split",   type=str, default="train")
    parser.add_argument("--field",   type=str, default="text",
                        help="텍스트 필드명")
    parser.add_argument("--lang",    type=str, default="en",
                        help="ko 설정 시 한국어 데이터셋 자동 선택")
    args = parser.parse_args()

    # 한국어 preset
    if args.lang == "ko" and not args.file:
        args.dataset = "oscar-corpus/OSCAR-2301"
        args.subset  = "ko"
        args.split   = "train"
        args.field   = "text"
        print("한국어 모드: OSCAR-2301 ko")

    enc = get_tokenizer()

    if args.file:
        process_file(args.file, enc)
    else:
        process_huggingface(
            args.dataset, args.subset, args.split, args.field, args.lang, enc)

    # vocab_size 저장 (C++에서 읽음)
    import json
    meta = {"vocab_size": enc.n_vocab, "tokenizer": "gpt2"}
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f)
    print(f"\nvocab_size={enc.n_vocab} saved to data/meta.json")
    print("\n학습 시작:")
    print("  config.txt에서 아래 설정 확인:")
    print("  DATASET_PATH=data/shard_0000.bin  (단일 파일)")
    print("  또는 DATASET_DIR=data  (shard 자동 탐색)")

if __name__ == "__main__":
    main()