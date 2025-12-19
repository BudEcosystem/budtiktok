use budtiktok_core::bpe_linear::{gpt2_pre_tokenize_fast, gpt2_pre_tokenize_simd};

fn main() {
    let test_cases = [
        "hello world",
        "hello world hello",
        "test",
    ];
    
    for text in &test_cases {
        let simd = gpt2_pre_tokenize_simd(text);
        let fast = gpt2_pre_tokenize_fast(text);
        
        println!("Input: {:?}", text);
        println!("  SIMD: {:?}", simd);
        println!("  Fast: {:?}", fast);
        println!("  Match: {}", simd == fast);
        println!();
    }
}
