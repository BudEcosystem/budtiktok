#!/usr/bin/env python3
"""Find exact point of divergence between HF and BT tokenization."""

import json
from transformers import BertTokenizerFast

DATA_PATH = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl"

def load_documents(path, limit=100):
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

    documents = load_documents(DATA_PATH, 50)

    # Get document 3
    doc_idx = 3
    doc = documents[doc_idx]

    hf_tokens = tokenizer.encode(doc, add_special_tokens=False)
    hf_token_strs = tokenizer.convert_ids_to_tokens(hf_tokens)
    bt_token_ids = bt_results["first_10_docs_tokens"][doc_idx]

    # Also get BT token strings by decoding IDs
    # The vocab is in the tokenizer, we can use HF's vocab to decode BT IDs
    bt_token_strs = [tokenizer.convert_ids_to_tokens([tid])[0] if tid < len(tokenizer) else f"[ID:{tid}]" for tid in bt_token_ids]

    print(f"\n=== Document {doc_idx} ===")
    print(f"HF tokens: {len(hf_tokens)}")
    print(f"BT tokens: {len(bt_token_ids)}")
    print(f"Difference: {len(bt_token_ids) - len(hf_tokens)}")

    # Find first divergence
    print("\n=== Finding first divergence ===")

    divergence_point = None
    for i in range(min(len(hf_token_strs), len(bt_token_strs))):
        if hf_token_strs[i] != bt_token_strs[i]:
            divergence_point = i
            print(f"\nFirst divergence at position {i}:")
            print(f"  HF tokens around divergence:")
            for j in range(max(0, i-5), min(len(hf_token_strs), i+10)):
                marker = " >>> " if j == i else "     "
                print(f"    {marker}{j}: {hf_token_strs[j]!r}")
            print(f"\n  BT tokens around divergence:")
            for j in range(max(0, i-5), min(len(bt_token_strs), i+10)):
                marker = " >>> " if j == i else "     "
                print(f"    {marker}{j}: {bt_token_strs[j]!r}")
            break

    if divergence_point is None:
        print("No divergence found in overlapping tokens!")
        print(f"HF has {len(hf_token_strs)} tokens, BT has {len(bt_token_strs)} tokens")
        if len(hf_token_strs) < len(bt_token_strs):
            print(f"\nBT extra tokens at end: {bt_token_strs[len(hf_token_strs):]}")
        else:
            print(f"\nHF extra tokens at end: {hf_token_strs[len(bt_token_strs):]}")

    # Now let's look at document 23 where HF has MORE tokens
    print("\n\n" + "="*80)
    doc_idx = 23
    doc = documents[doc_idx]

    hf_tokens = tokenizer.encode(doc, add_special_tokens=False)
    hf_token_strs = tokenizer.convert_ids_to_tokens(hf_tokens)
    bt_token_ids = bt_results["first_10_docs_tokens"][doc_idx] if doc_idx < len(bt_results["first_10_docs_tokens"]) else []

    print(f"\n=== Document {doc_idx} ===")
    print(f"HF tokens: {len(hf_tokens)}")

    if bt_token_ids:
        bt_token_strs = [tokenizer.convert_ids_to_tokens([tid])[0] if tid < len(tokenizer) else f"[ID:{tid}]" for tid in bt_token_ids]
        print(f"BT tokens: {len(bt_token_ids)}")
        print(f"Difference: {len(bt_token_ids) - len(hf_tokens)} (negative means HF has more)")

        # Find divergence
        print("\n=== Finding first divergence ===")
        for i in range(min(len(hf_token_strs), len(bt_token_strs))):
            if hf_token_strs[i] != bt_token_strs[i]:
                print(f"\nFirst divergence at position {i}:")
                print(f"  HF: {hf_token_strs[max(0,i-3):i+5]}")
                print(f"  BT: {bt_token_strs[max(0,i-3):i+5]}")
                break
    else:
        print("(Document 23 not in first 10 docs of BT results)")

if __name__ == "__main__":
    main()
