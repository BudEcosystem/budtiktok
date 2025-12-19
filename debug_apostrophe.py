#!/usr/bin/env python3
"""Debug apostrophe handling in HuggingFace vs BudTikTok."""

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Test curly apostrophe
test_cases = [
    "it's",         # straight apostrophe
    "it's",         # curly apostrophe (U+2019)
    "don't",        # straight
    "don't",        # curly
    "Russia's",     # straight
    "Russia's",     # curly
]

print("Testing apostrophe handling:\n")
for text in test_cases:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Input: {repr(text):20} -> Tokens: {token_strs} ({len(token_strs)} tokens)")

# Check what HF's normalizer does
print("\n\nTesting HF normalizer directly:")
print(f"Input: 'it\u2019s'")
backend = tokenizer.backend_tokenizer
norm = backend.normalizer
if norm:
    normalized = norm.normalize_str("it's")
    print(f"After normalize: {repr(normalized)}")
