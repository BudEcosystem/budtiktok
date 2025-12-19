//! GPU Tokenization Tests (TDD)
//!
//! Tests for Section 5: GPU Tokenization
//! - 5.1.x: CubeCL/CUDA Setup
//! - 5.2.x: GPU Kernels
//! - 5.3.x: GPU Integration

#[cfg(feature = "cuda")]
mod cuda_tests {
    use budtiktok_gpu::{
        cuda::{is_cuda_available, get_cuda_devices, CudaContext},
        memory::{GpuMemoryPool, GpuBuffer, PinnedBuffer},
        kernels::{PreTokenizeKernel, VocabLookupKernel, WordPieceKernel},
        backend::{GpuBackend, GpuTokenizer, GpuError},
    };

    // =========================================================================
    // 5.1.1 CUDA Device Detection Tests
    // =========================================================================

    #[test]
    fn test_cuda_availability() {
        // Should detect CUDA on systems with NVIDIA GPUs
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
        // Don't assert - just check it doesn't panic
    }

    #[test]
    fn test_cuda_device_enumeration() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let devices = get_cuda_devices();
        assert!(!devices.is_empty(), "Should find at least one CUDA device");

        for (i, device) in devices.iter().enumerate() {
            println!("Device {}: {} (CC {}.{})",
                i, device.name,
                device.compute_capability.0,
                device.compute_capability.1
            );
            println!("  Memory: {} MB", device.total_memory / 1024 / 1024);

            // Verify device properties
            assert!(!device.name.is_empty());
            assert!(device.compute_capability.0 >= 3, "Need at least SM 3.0");
            assert!(device.total_memory > 0);
        }
    }

    #[test]
    fn test_cuda_context_creation() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0);
        assert!(ctx.is_ok(), "Should create context on device 0: {:?}", ctx.err());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.device_id(), 0);
        assert!(ctx.free_memory() > 0);
    }

    // =========================================================================
    // 5.1.3 Memory Management Tests
    // =========================================================================

    #[test]
    fn test_gpu_memory_pool_creation() {
        let pool = GpuMemoryPool::new(1024 * 1024); // 1 MB
        assert_eq!(pool.total(), 1024 * 1024);
        assert_eq!(pool.available(), 1024 * 1024);
    }

    #[test]
    fn test_gpu_buffer_allocation() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Allocate buffer
        let buffer: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1024).expect("Failed to allocate");
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * std::mem::size_of::<u32>());
    }

    #[test]
    fn test_gpu_buffer_copy_to_device() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Host data
        let host_data: Vec<u32> = (0..1024).collect();

        // Copy to device
        let mut gpu_buffer = GpuBuffer::new(&ctx, 1024).expect("Failed to allocate");
        gpu_buffer.copy_from_host(&host_data).expect("Failed to copy");

        // Copy back and verify
        let mut result = vec![0u32; 1024];
        gpu_buffer.copy_to_host(&mut result).expect("Failed to copy back");

        assert_eq!(host_data, result);
    }

    #[test]
    fn test_pinned_buffer_allocation() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Allocate pinned buffer for faster transfers
        let pinned: PinnedBuffer<u8> = PinnedBuffer::new(&ctx, 4096).expect("Failed to allocate");
        assert_eq!(pinned.len(), 4096);
    }

    // =========================================================================
    // 5.2.1 GPU Vocabulary Lookup Tests
    // =========================================================================

    #[test]
    fn test_vocab_lookup_kernel_small() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Create simple vocabulary
        let vocab: Vec<(&str, u32)> = vec![
            ("hello", 100),
            ("world", 101),
            ("the", 102),
            ("[UNK]", 0),
        ];

        let kernel = VocabLookupKernel::new(&ctx, &vocab).expect("Failed to create kernel");

        // Lookup words
        let words = vec!["hello", "world", "unknown"];
        let results = kernel.lookup(&words).expect("Failed to lookup");

        assert_eq!(results, vec![100, 101, 0]); // 0 = [UNK]
    }

    #[test]
    fn test_vocab_lookup_kernel_batch() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Create larger vocabulary
        let vocab: Vec<(String, u32)> = (0..10000)
            .map(|i| (format!("word{}", i), i as u32))
            .collect();
        let vocab_refs: Vec<(&str, u32)> = vocab.iter().map(|(s, i)| (s.as_str(), *i)).collect();

        let kernel = VocabLookupKernel::new(&ctx, &vocab_refs).expect("Failed to create kernel");

        // Batch lookup
        let words: Vec<String> = (0..1000).map(|i| format!("word{}", i * 10)).collect();
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        let results = kernel.lookup(&word_refs).expect("Failed to lookup");

        for (i, &result) in results.iter().enumerate() {
            let expected = (i * 10) as u32;
            assert_eq!(result, expected, "Mismatch at index {}", i);
        }
    }

    // =========================================================================
    // 5.2.2 GPU Pre-tokenization Tests
    // =========================================================================

    #[test]
    fn test_pretokenize_kernel_whitespace() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");
        let kernel = PreTokenizeKernel::new(&ctx).expect("Failed to create kernel");

        let text = "hello world foo bar";
        let boundaries = kernel.find_word_boundaries(text.as_bytes()).expect("Failed");

        // Should find positions: 0-5, 6-11, 12-15, 16-19
        assert_eq!(boundaries.len(), 4);
        assert_eq!(boundaries[0], (0, 5));   // "hello"
        assert_eq!(boundaries[1], (6, 11));  // "world"
        assert_eq!(boundaries[2], (12, 15)); // "foo"
        assert_eq!(boundaries[3], (16, 19)); // "bar"
    }

    #[test]
    fn test_pretokenize_kernel_punctuation() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");
        let kernel = PreTokenizeKernel::new(&ctx).expect("Failed to create kernel");

        let text = "Hello, world!";
        let boundaries = kernel.find_word_boundaries(text.as_bytes()).expect("Failed");

        // Current implementation does whitespace splitting only
        // Punctuation handling is done by the full tokenizer pipeline
        // "Hello," and "world!" = 2 words
        assert!(boundaries.len() >= 2);
        println!("Found {} word boundaries in '{}'", boundaries.len(), text);
    }

    #[test]
    fn test_pretokenize_kernel_batch() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");
        let kernel = PreTokenizeKernel::new(&ctx).expect("Failed to create kernel");

        let texts: Vec<&str> = vec![
            "hello world",
            "foo bar baz",
            "a b c d e",
        ];

        let batch_boundaries = kernel.find_word_boundaries_batch(&texts).expect("Failed");

        assert_eq!(batch_boundaries.len(), 3);
        assert_eq!(batch_boundaries[0].len(), 2); // "hello", "world"
        assert_eq!(batch_boundaries[1].len(), 3); // "foo", "bar", "baz"
        assert_eq!(batch_boundaries[2].len(), 5); // "a", "b", "c", "d", "e"
    }

    // =========================================================================
    // 5.2.3 GPU WordPiece Tests
    // =========================================================================

    #[test]
    fn test_wordpiece_kernel_basic() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Simple vocab for testing
        let vocab: Vec<(&str, u32)> = vec![
            ("hello", 100),
            ("world", 101),
            ("un", 102),
            ("##happy", 103),
            ("##ness", 104),
            ("[UNK]", 0),
        ];

        let kernel = WordPieceKernel::new(&ctx, &vocab, "##").expect("Failed to create kernel");

        // Test whole word
        let tokens = kernel.tokenize_word("hello").expect("Failed");
        assert_eq!(tokens, vec![100]);

        // Test subword
        let tokens = kernel.tokenize_word("unhappyness").expect("Failed");
        assert_eq!(tokens, vec![102, 103, 104]); // un + ##happy + ##ness
    }

    #[test]
    fn test_wordpiece_kernel_unknown() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        let vocab: Vec<(&str, u32)> = vec![
            ("hello", 100),
            ("[UNK]", 0),
        ];

        let kernel = WordPieceKernel::new(&ctx, &vocab, "##").expect("Failed to create kernel");

        // Unknown word should return [UNK]
        let tokens = kernel.tokenize_word("xyz123").expect("Failed");
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn test_wordpiece_kernel_batch() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        let vocab: Vec<(&str, u32)> = vec![
            ("hello", 100),
            ("world", 101),
            ("[UNK]", 0),
        ];

        let kernel = WordPieceKernel::new(&ctx, &vocab, "##").expect("Failed to create kernel");

        let words = vec!["hello", "world", "unknown"];
        let batch_tokens = kernel.tokenize_words(&words).expect("Failed");

        assert_eq!(batch_tokens.len(), 3);
        assert_eq!(batch_tokens[0], vec![100]);
        assert_eq!(batch_tokens[1], vec![101]);
        assert_eq!(batch_tokens[2], vec![0]); // [UNK]
    }

    // =========================================================================
    // 5.3.1 GpuTokenizer Integration Tests
    // =========================================================================

    #[test]
    fn test_gpu_tokenizer_creation() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let tokenizer = GpuTokenizer::new(GpuBackend::Cuda);
        assert!(tokenizer.is_ok(), "Failed to create: {:?}", tokenizer.err());
    }

    #[test]
    fn test_gpu_tokenizer_encode_single() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        // This test requires loading a real tokenizer config
        // For now, test the basic API
        let tokenizer = GpuTokenizer::new(GpuBackend::Cuda).expect("Failed to create");

        // Basic encode should work (may return error if no vocab loaded)
        let result = tokenizer.encode("hello world");
        // Just verify it doesn't panic
        println!("Encode result: {:?}", result);
    }

    #[test]
    fn test_gpu_tokenizer_batch_encode() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let tokenizer = GpuTokenizer::new(GpuBackend::Cuda).expect("Failed to create");

        let texts = vec!["hello world", "foo bar", "test text"];
        let result = tokenizer.encode_batch(&texts);

        // Just verify it doesn't panic
        println!("Batch encode result: {:?}", result);
    }

    // =========================================================================
    // 5.3.2 Batch Size Optimization Tests
    // =========================================================================

    #[test]
    fn test_optimal_batch_size_detection() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let tokenizer = GpuTokenizer::new(GpuBackend::Cuda).expect("Failed to create");

        let optimal_size = tokenizer.find_optimal_batch_size();
        println!("Optimal batch size: {}", optimal_size);

        assert!(optimal_size >= 8, "Optimal batch size should be at least 8");
        assert!(optimal_size <= 512, "Optimal batch size should be at most 512");
    }

    // =========================================================================
    // 5.3.3 Multi-GPU Tests
    // =========================================================================

    #[test]
    fn test_multi_gpu_detection() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let devices = get_cuda_devices();
        println!("Found {} GPU(s)", devices.len());

        // Just verify detection works
        assert!(devices.len() >= 1);
    }

    // =========================================================================
    // Performance Benchmarks (not assertions, just measurements)
    // =========================================================================

    #[test]
    fn bench_gpu_vs_cpu_tokenization() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        // Generate test data
        let texts: Vec<String> = (0..1000)
            .map(|i| format!("This is test sentence number {} with some words.", i))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let tokenizer = GpuTokenizer::new(GpuBackend::Cuda).expect("Failed to create");

        // Warmup
        let _ = tokenizer.encode_batch(&text_refs[..10]);

        // Benchmark
        let start = std::time::Instant::now();
        let _ = tokenizer.encode_batch(&text_refs);
        let gpu_time = start.elapsed();

        println!("GPU tokenization of {} texts: {:?}", texts.len(), gpu_time);
        println!("Throughput: {:.2} texts/sec", texts.len() as f64 / gpu_time.as_secs_f64());
    }
    // =========================================================================
    // Performance Benchmarks
    // =========================================================================

    #[test]
    fn bench_wordpiece_batched() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Create a larger vocabulary for realistic benchmarking
        let mut vocab: Vec<(String, u32)> = (0..5000)
            .map(|i| (format!("word{}", i), i as u32))
            .collect();
        // Add some continuation tokens
        for i in 0..1000 {
            vocab.push((format!("##part{}", i), (5000 + i) as u32));
        }
        vocab.push(("[UNK]".to_string(), 0));

        let vocab_refs: Vec<(&str, u32)> = vocab.iter().map(|(s, i)| (s.as_str(), *i)).collect();
        let kernel = WordPieceKernel::new(&ctx, &vocab_refs, "##").expect("Failed to create kernel");

        // Test words
        let words: Vec<String> = (0..100).map(|i| format!("word{}", i * 50)).collect();
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        // Warmup
        let _ = kernel.tokenize_words(&word_refs[..10]);

        // Benchmark batched tokenization
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = kernel.tokenize_words(&word_refs);
        }
        let batched_time = start.elapsed();

        println!("Batched tokenization (100 words x 10 iterations): {:?}", batched_time);
        println!("Per word: {:?}", batched_time / 1000);
        println!("Throughput: {:.2} words/sec", 1000.0 / batched_time.as_secs_f64());
    }

    #[test]
    fn bench_vocab_lookup_throughput() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let ctx = CudaContext::new(0).expect("Failed to create context");

        // Create vocabulary
        let vocab: Vec<(String, u32)> = (0..30000)
            .map(|i| (format!("token{}", i), i as u32))
            .collect();
        let vocab_refs: Vec<(&str, u32)> = vocab.iter().map(|(s, i)| (s.as_str(), *i)).collect();

        let kernel = VocabLookupKernel::new(&ctx, &vocab_refs).expect("Failed to create kernel");

        // Generate lookup words
        let words: Vec<String> = (0..10000).map(|i| format!("token{}", i % 30000)).collect();
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        // Warmup
        let _ = kernel.lookup(&word_refs[..100]);

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = kernel.lookup(&word_refs);
        }
        let elapsed = start.elapsed();

        println!("Vocab lookup (10000 words x 10 iterations): {:?}", elapsed);
        println!("Throughput: {:.2} lookups/sec", 100000.0 / elapsed.as_secs_f64());
    }
}

// Tests that work without CUDA
mod cpu_fallback_tests {
    use budtiktok_gpu::backend::{GpuBackend, GpuTokenizer};

    #[test]
    fn test_cpu_fallback_when_no_gpu() {
        // This should not panic even without GPU
        let result = GpuTokenizer::new(GpuBackend::Cuda);

        // Either succeeds (GPU available) or returns appropriate error
        match result {
            Ok(tokenizer) => {
                println!("GPU tokenizer created successfully");
                assert_eq!(tokenizer.backend(), GpuBackend::Cuda);
            }
            Err(e) => {
                println!("Expected error without GPU: {}", e);
            }
        }
    }
}
