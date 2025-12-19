use budtiktok_core::memory::StringInterner;

#[test]
fn test_interner_unsoundness() {
    let s = {
        let interner = StringInterner::new();
        interner.intern("hello world")
    };
    // interner is dropped here.
    // s is accessed here.
    // This should crash or show garbage if the bug exists.
    println!("String: {}", s);
}
