#!/usr/bin/env python3
"""Find remaining tokenization differences."""

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

    # Load BudTikTok results
    with open("/home/bud/Desktop/latentbud/budtiktok/wordpiece_budtiktok_results.json") as f:
        bt_results = json.load(f)

    documents = load_documents(DATA_PATH, 500)

    # Find document 20 (first mismatch)
    for doc_idx in [20, 326, 331]:
        if doc_idx >= len(documents):
            continue

        doc = documents[doc_idx]
        hf_tokens = tokenizer.encode(doc, add_special_tokens=False)
        hf_token_strs = tokenizer.convert_ids_to_tokens(hf_tokens)
        bt_count = bt_results["token_counts"][doc_idx]

        print(f"\n{'='*80}")
        print(f"Document {doc_idx}: HF={len(hf_tokens)}, BT={bt_count}, diff={bt_count - len(hf_tokens)}")
        print(f"{'='*80}")

        # Find unusual characters
        print("\nUnusual characters (first 20):")
        unusual_count = 0
        for i, c in enumerate(doc):
            if ord(c) > 127 or c in '$€£¥@#%^&*':
                context_start = max(0, i-5)
                context_end = min(len(doc), i+15)
                print(f"  pos {i}: U+{ord(c):04X} {repr(c):8} | context: {repr(doc[context_start:context_end])}")
                unusual_count += 1
                if unusual_count >= 20:
                    print("  ... (more)")
                    break

        # Get pre-tokenized output from HF
        backend = tokenizer.backend_tokenizer
        pre_tokenizer = backend.pre_tokenizer
        if pre_tokenizer:
            # Get first 200 chars pre-tokenized
            sample = doc[:500]
            pre_tokens = pre_tokenizer.pre_tokenize_str(sample)
            print(f"\nHF pre-tokenized sample (first 30 tokens):")
            for i, (tok, span) in enumerate(pre_tokens[:30]):
                print(f"  {i:3d}: {repr(tok):20} span={span}")

if __name__ == "__main__":
    main()
