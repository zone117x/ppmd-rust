//! Regression tests for reported security issues.

use std::io::{Read, Write};

use ppmd_rust::{Ppmd8Decoder, Ppmd8Encoder, RestoreMethod};

/// Deterministic pseudo natural language text.
///
/// A xorshift generator picks words from a fixed vocabulary, so the corpus is large
/// enough and repetitive enough to fill a 1 MiB model and force the model restoration
/// path of PPMd8.
fn pseudo_text(size: usize) -> Vec<u8> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let vocabulary: Vec<String> = (0..60_000)
        .map(|_| {
            let length = 3 + (next() % 7) as usize;
            (0..length)
                .map(|_| (b'a' + (next() % 26) as u8) as char)
                .collect()
        })
        .collect();

    let mut text = Vec::with_capacity(size + 16);
    while text.len() < size {
        let word = &vocabulary[(next() % vocabulary.len() as u64) as usize];
        text.extend_from_slice(word.as_bytes());
        text.push(b' ');
    }

    text
}

/// GHSA-rqc2-j9v2-j22v: out of bounds heap reads and writes in the PPMd8 decoder.
///
/// A ZIP entry that uses compression method 98 carries the order, the memory size and
/// the restoration method in a 2 byte parameter word that the producer chooses freely.
/// A decoder therefore has to survive a stream whose restoration method does not match
/// the one that the encoder used. The model then diverges, the arena runs full and
/// `restore_model()` enters `cut_off()`, where the model bookkeeping is inconsistent.
///
/// Decoding such a stream may report an error or return garbage, but it must stay
/// inside the model arena.
#[test]
fn ghsa_rqc2_j9v2_j22v_ppmd8_cut_off_out_of_bounds() {
    const ORDER: u32 = 16;
    const MEMORY_SIZE: u32 = 1 << 20;

    let text = pseudo_text(512 << 10);

    let mut compressed = Vec::new();
    {
        let mut encoder =
            Ppmd8Encoder::new(&mut compressed, ORDER, MEMORY_SIZE, RestoreMethod::Restart).unwrap();
        encoder.write_all(&text).unwrap();
        encoder.finish(true).unwrap();
    }

    // Same stream, but the restoration method of the parameter word is flipped to CutOff.
    let mut decoder = Ppmd8Decoder::new(
        compressed.as_slice(),
        ORDER,
        MEMORY_SIZE,
        RestoreMethod::CutOff,
    )
    .unwrap();

    let mut decompressed = Vec::new();
    let _ = decoder.read_to_end(&mut decompressed);
}

/// GHSA-rqc2-j9v2-j22v: the model restoration path of PPMd8.
///
/// [`RestoreMethod::CutOff`] keeps the model and reduces it instead of restarting it.
/// The text is much larger than the model memory, so `restore_model()` runs many times
/// and walks the whole reduction path.
#[test]
fn ppmd8_cut_off_round_trip() {
    const ORDER: u32 = 16;
    const MEMORY_SIZE: u32 = 1 << 20;

    let text = pseudo_text(4 * 1024 * 1024);

    let mut compressed = Vec::new();
    {
        let mut encoder =
            Ppmd8Encoder::new(&mut compressed, ORDER, MEMORY_SIZE, RestoreMethod::CutOff).unwrap();
        encoder.write_all(&text).unwrap();
        encoder.finish(true).unwrap();
    }

    let mut decompressed = Vec::new();
    {
        let mut decoder = Ppmd8Decoder::new(
            compressed.as_slice(),
            ORDER,
            MEMORY_SIZE,
            RestoreMethod::CutOff,
        )
        .unwrap();
        decoder.read_to_end(&mut decompressed).unwrap();
    }

    assert_eq!(decompressed, text);
}

#[test]
fn ppmd7a_restarts_the_range_coder_and_keeps_the_model() {
    use std::io::{Read, Write};

    use ppmd_rust::{Ppmd7aDecoder, Ppmd7aEncoder};

    let first = b"the first block of text, repeated words words words words";
    let second = b"the second block, with the same words words words again";

    let mut compressed = Vec::new();
    let mut encoder = Ppmd7aEncoder::new(&mut compressed, 6, 1 << 20).unwrap();
    encoder.write_all(first).unwrap();
    encoder.restart_range_coder().unwrap();
    let after_first = encoder.get_ref().len();
    encoder.write_all(second).unwrap();
    encoder.finish(false).unwrap();

    let mut decoder = Ppmd7aDecoder::new(&compressed[..], 6, 1 << 20).unwrap();
    let mut out = vec![0u8; first.len()];
    decoder.read_exact(&mut out).unwrap();
    assert_eq!(&out, first);
    // The first coder's flushed bytes were consumed exactly; the second block
    // starts where the encoder restarted.
    assert_eq!(compressed.len() - decoder.get_ref().len(), after_first);
    decoder.restart_range_coder().unwrap();
    let mut out = vec![0u8; second.len()];
    decoder.read_exact(&mut out).unwrap();
    assert_eq!(&out, second);
}
