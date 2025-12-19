//! Fuzzing Target: IPC Deserialization (9.5.3)
//!
//! Fuzzes serialized message deserialization to ensure no panics
//! on malformed binary data.
//!
//! Run with: cargo +nightly fuzz run fuzz_ipc

#![no_main]

use libfuzzer_sys::fuzz_target;
use budtiktok_core::Encoding;

fuzz_target!(|data: &[u8]| {
    // Fuzz various binary deserialization paths
    fuzz_encoding_binary(data);
    fuzz_length_prefixed(data);
    fuzz_token_ids(data);
});

/// Fuzz binary encoding data
fn fuzz_encoding_binary(data: &[u8]) {
    // Try to interpret data as various encoding structures

    // Interpret as length-prefixed token IDs
    if data.len() >= 4 {
        let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        // Don't allocate insanely large buffers
        if len <= 1_000_000 && data.len() >= 4 + len * 4 {
            let mut encoding = Encoding::new();

            for i in 0..len.min(100_000) {
                let offset = 4 + i * 4;
                if offset + 4 <= data.len() {
                    let id = u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    encoding.push(id, format!("t{}", id), (0, 1), Some(i as u32), Some(0), false);
                }
            }

            // Test operations on the constructed encoding
            let _ = encoding.len();
            let _ = encoding.get_ids();
            let _ = encoding.get_attention_mask();
        }
    }
}

/// Fuzz length-prefixed message format
fn fuzz_length_prefixed(data: &[u8]) {
    let mut offset = 0;

    // Try to parse multiple length-prefixed messages
    while offset + 4 <= data.len() {
        let msg_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;

        // Sanity check length
        if msg_len > 10_000_000 {
            break;
        }

        offset += 4;

        if offset + msg_len > data.len() {
            break;
        }

        // Try to interpret message as UTF-8 text
        if let Ok(text) = std::str::from_utf8(&data[offset..offset + msg_len]) {
            // Successfully parsed as text, could be a tokenization request
            let _ = text.len();
        }

        offset += msg_len;
    }
}

/// Fuzz raw token ID sequences
fn fuzz_token_ids(data: &[u8]) {
    // Interpret data as packed u32 token IDs
    let mut ids = Vec::new();

    for chunk in data.chunks_exact(4) {
        let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        ids.push(id);

        // Limit to prevent memory exhaustion
        if ids.len() > 100_000 {
            break;
        }
    }

    // Build an encoding from the IDs
    let mut encoding = Encoding::new();
    for (i, &id) in ids.iter().enumerate() {
        encoding.push(id, format!("t{}", id), (i, i + 1), Some(i as u32), Some(0), false);
    }

    // Test various encoding operations
    if encoding.len() > 10 {
        let mut truncated = encoding.clone();
        truncated.truncate(encoding.len() / 2, 0);
    }

    let mut padded = encoding.clone();
    padded.pad(encoding.len() + 10, 0, "[PAD]");

    // Test attention mask
    let mask = encoding.get_attention_mask();
    assert_eq!(mask.len(), encoding.len());

    // Test type IDs
    let type_ids = encoding.get_type_ids();
    assert_eq!(type_ids.len(), encoding.len());
}
