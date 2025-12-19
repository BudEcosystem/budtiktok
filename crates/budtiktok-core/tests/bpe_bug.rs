use budtiktok_core::bpe::{LinearBpeEncoder, VocabAutomaton, CompatibilityTable, MergeRule};
use budtiktok_core::vocab::VocabularyBuilder;

#[test]
fn test_linear_bpe_bug() {
    // Create a vocab with "hello" and "world" but NO merge rule for them.
    let vocab = VocabularyBuilder::new()
        .add_tokens(["[UNK]", "hello", "world"])
        .unk_token("[UNK]")
        .build();

    let merges = vec![]; // No merges
    let compat = CompatibilityTable::from_merges(&merges, &vocab);
    let automaton = VocabAutomaton::from_vocab(&vocab);
    
    let encoder = LinearBpeEncoder::new(automaton, compat, 0);

    // "hello world" should be encoded as ["hello", "world"] (assuming pre-tokenization splits them or we pass them directly?)
    // Wait, LinearBpeEncoder encodes a string.
    // If we pass "helloworld" (no space), and "hello", "world" are tokens.
    // It should find "hello", "world".
    // But they are not compatible (no merge).
    // So the code might fail.
    
    let ids = encoder.encode("helloworld");
    println!("IDs: {:?}", ids);
    
    // IDs should be [1, 2] (hello, world)
    assert_eq!(ids.len(), 2);
}
