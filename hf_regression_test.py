#!/usr/bin/env python3
"""
HuggingFace Tokenizer Regression Test

Runs HuggingFace tokenizers on the 1GB dataset and compares with BudTikTok results.
Tests both GPT-2 (BPE) and BERT (WordPiece) tokenizers.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count

# HuggingFace tokenizers
from transformers import GPT2TokenizerFast, BertTokenizerFast

WORKSPACE = "/home/bud/Desktop/latentbud/budtiktok"
DATA_PATH = f"{WORKSPACE}/benchmark_data/openwebtext_1gb.jsonl"
NUM_DOCS = 10000  # Match Rust test

def load_documents(path: str, limit: int = NUM_DOCS) -> List[str]:
    """Load documents from JSONL file."""
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

def encode_single_gpt2(text: str) -> List[int]:
    """Encode single text with GPT-2 tokenizer (for multiprocessing)."""
    global gpt2_tokenizer
    return gpt2_tokenizer.encode(text)

def encode_single_bert(text: str) -> List[int]:
    """Encode single text with BERT tokenizer (for multiprocessing)."""
    global bert_tokenizer
    return bert_tokenizer.encode(text, add_special_tokens=False)

def init_gpt2_worker():
    """Initialize GPT-2 tokenizer for worker process."""
    global gpt2_tokenizer
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def init_bert_worker():
    """Initialize BERT tokenizer for worker process."""
    global bert_tokenizer
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def run_gpt2_tests(documents: List[str]) -> Dict[str, Any]:
    """Run GPT-2 (BPE) tokenizer tests."""
    print("\n" + "━" * 80)
    print("                          GPT-2 (BPE) TOKENIZER TESTS")
    print("━" * 80 + "\n")

    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    total_bytes = sum(len(d.encode('utf-8')) for d in documents)
    print(f"Documents: {len(documents)}")
    print(f"Total size: {total_bytes / 1024 / 1024:.2f} MB\n")

    # Single-core test
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ TEST 1: GPT-2 Single-Core Encoding                                         │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    start = time.time()
    single_results = []
    total_tokens = 0
    for doc in documents:
        tokens = tokenizer.encode(doc)
        single_results.append(tokens)
        total_tokens += len(tokens)

    single_time = time.time() - start
    single_throughput = total_bytes / single_time / 1024 / 1024

    print(f"  Documents:    {len(documents)}")
    print(f"  Tokens:       {total_tokens}")
    print(f"  Time:         {single_time:.2f}s")
    print(f"  Throughput:   {single_throughput:.2f} MB/s")
    print(f"  Tokens/sec:   {total_tokens / single_time:.0f}")
    print()

    # Multi-core test
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ TEST 2: GPT-2 Multi-Core Encoding                                          │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    num_workers = cpu_count()
    start = time.time()

    with Pool(num_workers, initializer=init_gpt2_worker) as pool:
        multi_results = pool.map(encode_single_gpt2, documents)

    multi_time = time.time() - start
    multi_tokens = sum(len(r) for r in multi_results)
    multi_throughput = total_bytes / multi_time / 1024 / 1024

    print(f"  Workers:      {num_workers}")
    print(f"  Documents:    {len(documents)}")
    print(f"  Tokens:       {multi_tokens}")
    print(f"  Time:         {multi_time:.2f}s")
    print(f"  Throughput:   {multi_throughput:.2f} MB/s")
    print(f"  Tokens/sec:   {multi_tokens / multi_time:.0f}")
    print(f"  Speedup:      {single_time / multi_time:.2f}x over single-core")
    print()

    # Verify consistency
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ TEST 3: GPT-2 Consistency Check                                            │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    mismatches = 0
    for i, (single, multi) in enumerate(zip(single_results, multi_results)):
        if single != multi:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH at doc {i}: single={len(single)} tokens, multi={len(multi)} tokens")

    if mismatches == 0:
        print("  ✓ PASS: Single-core and multi-core produce identical results")
    else:
        print(f"  ✗ FAIL: {mismatches} mismatches found")
    print()

    return {
        "total_tokens": total_tokens,
        "token_counts": [len(r) for r in single_results],
        "first_10_docs_tokens": [r for r in single_results[:10]],
        "single_core_time": single_time,
        "single_core_throughput": single_throughput,
        "multi_core_time": multi_time,
        "multi_core_throughput": multi_throughput,
    }

def run_bert_tests(documents: List[str]) -> Dict[str, Any]:
    """Run BERT (WordPiece) tokenizer tests."""
    print("\n" + "━" * 80)
    print("                        BERT (WORDPIECE) TOKENIZER TESTS")
    print("━" * 80 + "\n")

    # Load tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    total_bytes = sum(len(d.encode('utf-8')) for d in documents)
    print(f"Documents: {len(documents)}")
    print(f"Total size: {total_bytes / 1024 / 1024:.2f} MB\n")

    # Single-core test
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ TEST 1: BERT Single-Core Encoding                                          │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    start = time.time()
    single_results = []
    total_tokens = 0
    for doc in documents:
        # Don't add special tokens for fair comparison
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        single_results.append(tokens)
        total_tokens += len(tokens)

    single_time = time.time() - start
    single_throughput = total_bytes / single_time / 1024 / 1024

    print(f"  Documents:    {len(documents)}")
    print(f"  Tokens:       {total_tokens}")
    print(f"  Time:         {single_time:.2f}s")
    print(f"  Throughput:   {single_throughput:.2f} MB/s")
    print(f"  Tokens/sec:   {total_tokens / single_time:.0f}")
    print()

    # Multi-core test
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ TEST 2: BERT Multi-Core Encoding                                           │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    num_workers = cpu_count()
    start = time.time()

    with Pool(num_workers, initializer=init_bert_worker) as pool:
        multi_results = pool.map(encode_single_bert, documents)

    multi_time = time.time() - start
    multi_tokens = sum(len(r) for r in multi_results)
    multi_throughput = total_bytes / multi_time / 1024 / 1024

    print(f"  Workers:      {num_workers}")
    print(f"  Documents:    {len(documents)}")
    print(f"  Tokens:       {multi_tokens}")
    print(f"  Time:         {multi_time:.2f}s")
    print(f"  Throughput:   {multi_throughput:.2f} MB/s")
    print(f"  Tokens/sec:   {multi_tokens / multi_time:.0f}")
    print(f"  Speedup:      {single_time / multi_time:.2f}x over single-core")
    print()

    return {
        "total_tokens": total_tokens,
        "token_counts": [len(r) for r in single_results],
        "first_10_docs_tokens": [r for r in single_results[:10]],
        "single_core_time": single_time,
        "single_core_throughput": single_throughput,
        "multi_core_time": multi_time,
        "multi_core_throughput": multi_throughput,
    }

def compare_with_budtiktok(hf_results: Dict[str, Any], budtiktok_path: str, name: str) -> bool:
    """Compare HuggingFace results with BudTikTok results."""
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ ACCURACY CHECK: {name} BudTikTok vs HuggingFace" + " " * (55 - len(name)) + "│")
    print(f"└─────────────────────────────────────────────────────────────────────────────┘")

    if not os.path.exists(budtiktok_path):
        print(f"  ⚠ BudTikTok results not found: {budtiktok_path}")
        print("  Run the Rust regression test first.")
        return False

    with open(budtiktok_path, 'r') as f:
        bt_results = json.load(f)

    # Compare total tokens
    hf_total = hf_results["total_tokens"]
    bt_total = bt_results["total_tokens"]

    print(f"  HuggingFace total tokens: {hf_total:,}")
    print(f"  BudTikTok total tokens:   {bt_total:,}")

    if hf_total == bt_total:
        print(f"  ✓ Total tokens match exactly!")
    else:
        diff = abs(hf_total - bt_total)
        pct = diff / hf_total * 100
        print(f"  ✗ Token count difference: {diff:,} ({pct:.4f}%)")

    # Compare per-document token counts
    hf_counts = hf_results["token_counts"]
    bt_counts = bt_results["token_counts"]

    mismatches = 0
    mismatch_details = []
    for i, (hf, bt) in enumerate(zip(hf_counts, bt_counts)):
        if hf != bt:
            mismatches += 1
            if len(mismatch_details) < 5:
                mismatch_details.append((i, hf, bt))

    if mismatches == 0:
        print(f"  ✓ All {len(hf_counts)} documents have matching token counts!")
        return True
    else:
        print(f"  ✗ {mismatches} documents have mismatching token counts:")
        for doc_id, hf, bt in mismatch_details:
            print(f"      Doc {doc_id}: HF={hf}, BT={bt}, diff={hf-bt}")
        return False

def main():
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║         HUGGINGFACE TOKENIZER REGRESSION TEST - BPE & WORDPIECE             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    # Load documents
    print(f"\nLoading documents from: {DATA_PATH}")
    documents = load_documents(DATA_PATH, NUM_DOCS)
    print(f"Loaded {len(documents)} documents")

    # Run GPT-2 (BPE) tests
    gpt2_results = run_gpt2_tests(documents)

    # Save GPT-2 results
    gpt2_output = f"{WORKSPACE}/bpe_hf_results.json"
    with open(gpt2_output, 'w') as f:
        json.dump(gpt2_results, f, indent=2)
    print(f"  Saved GPT-2 results to: {gpt2_output}")

    # Run BERT (WordPiece) tests
    bert_results = run_bert_tests(documents)

    # Save BERT results
    bert_output = f"{WORKSPACE}/wordpiece_hf_results.json"
    with open(bert_output, 'w') as f:
        json.dump(bert_results, f, indent=2)
    print(f"  Saved BERT results to: {bert_output}")

    # Compare with BudTikTok results
    print("\n" + "═" * 80)
    print("                         ACCURACY COMPARISON")
    print("═" * 80)

    bpe_match = compare_with_budtiktok(
        gpt2_results,
        f"{WORKSPACE}/bpe_budtiktok_results.json",
        "BPE"
    )

    wp_match = compare_with_budtiktok(
        bert_results,
        f"{WORKSPACE}/wordpiece_budtiktok_results.json",
        "WordPiece"
    )

    # Final summary
    print("\n" + "═" * 80)
    print("                         FINAL SUMMARY")
    print("═" * 80)

    print("\n  ┌────────────────────────────────────────────────────────────────────────┐")
    print("  │                    PERFORMANCE COMPARISON                              │")
    print("  ├────────────────────────────────────────────────────────────────────────┤")
    print("  │  Tokenizer    │ Mode         │ HF (MB/s)  │ Note                       │")
    print("  ├───────────────┼──────────────┼────────────┼────────────────────────────┤")
    print(f"  │  GPT-2 (BPE)  │ Single-core  │ {gpt2_results['single_core_throughput']:>8.2f}   │ Baseline                   │")
    print(f"  │  GPT-2 (BPE)  │ Multi-core   │ {gpt2_results['multi_core_throughput']:>8.2f}   │ {cpu_count()} workers              │")
    print(f"  │  BERT (WP)    │ Single-core  │ {bert_results['single_core_throughput']:>8.2f}   │ Baseline                   │")
    print(f"  │  BERT (WP)    │ Multi-core   │ {bert_results['multi_core_throughput']:>8.2f}   │ {cpu_count()} workers              │")
    print("  └───────────────┴──────────────┴────────────┴────────────────────────────┘")

    print("\n  ┌────────────────────────────────────────────────────────────────────────┐")
    print("  │                    ACCURACY VERIFICATION                               │")
    print("  ├────────────────────────────────────────────────────────────────────────┤")
    bpe_status = "✓ 100% MATCH" if bpe_match else "✗ MISMATCH"
    wp_status = "✓ 100% MATCH" if wp_match else "✗ MISMATCH" if not wp_match else "⚠ NOT TESTED"
    print(f"  │  BPE (GPT-2)      │ {bpe_status:<20} │ vs HuggingFace GPT2TokenizerFast │")
    print(f"  │  WordPiece (BERT) │ {wp_status:<20} │ vs HuggingFace BertTokenizerFast │")
    print("  └───────────────────┴──────────────────────┴──────────────────────────────┘")

    print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    HUGGINGFACE REGRESSION TEST COMPLETE                      ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
