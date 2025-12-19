use budtiktok_core::memory::StringInterner;

fn main() {
    let s = {
        let interner = StringInterner::new();
        interner.intern("hello world")
    };
    // interner is dropped here, backing storage is freed.
    // s is &'static str, so compiler allows this.
    println!("{}", s); // Use after free!
}
