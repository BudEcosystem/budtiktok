#!/usr/bin/env python3
"""Investigate final remaining differences."""

import json
from transformers import BertTokenizerFast

DATA_PATH = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl"

def load_documents(path, limit=10000):
    documents = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                data = json.loads(line)
                if 'text' in data:
                    documents.append(data['text'])
            except json.JSONDecodeError:
                continue
    return documents

def main():
    print("Loading HuggingFace BERT tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    backend = tokenizer.backend_tokenizer
    pre_tokenizer = backend.pre_tokenizer

    documents = load_documents(DATA_PATH, 2000)

    # Check document 1489 which has diff=18 (largest visible diff)
    for doc_idx in [603, 1489]:
        if doc_idx >= len(documents):
            continue

        doc = documents[doc_idx]

        print(f"\n{'='*80}")
        print(f"Document {doc_idx}")
        print(f"{'='*80}")

        hf_tokens = tokenizer.encode(doc, add_special_tokens=False)
        hf_strs = tokenizer.convert_ids_to_tokens(hf_tokens)

        print(f"Length: {len(doc)} chars")
        print(f"HF tokens: {len(hf_tokens)}")
        print(f"\nFirst 500 chars:")
        print(repr(doc[:500]))

        # Find unusual characters
        print(f"\nUnusual characters:")
        for i, c in enumerate(doc):
            if ord(c) > 127:
                context_start = max(0, i-10)
                context_end = min(len(doc), i+20)
                print(f"  pos {i}: U+{ord(c):04X} {repr(c):8} | {repr(doc[context_start:context_end])}")
                if i > 300:
                    break

        # Pre-tokenize
        pre_tokens = pre_tokenizer.pre_tokenize_str(doc[:500]) if pre_tokenizer else []
        print(f"\nHF pre-tokenized (first 50):")
        for i, (tok, span) in enumerate(pre_tokens[:50]):
            print(f"  {i:3d}: {repr(tok):25} span={span}")

        # Actual tokens
        print(f"\nHF tokens (first 50):")
        for i, tok in enumerate(hf_strs[:50]):
            print(f"  {i:3d}: {repr(tok)}")

if __name__ == "__main__":
    main()
