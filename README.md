# NEFS — New Engineered Featural Script

**NEFS** is a binary phonetic encoding system designed as a principled replacement for the International Phonetic Alphabet (IPA). Where IPA is a writing system that grew historically — symbols added piecemeal over more than a century — NEFS is engineered from the ground up with two goals: a more systematic organization of phonetic features, and a representation that is fast and parallelizable on modern hardware.

---

## Why Not Just Use IPA?

IPA is the gold standard for human-readable phonetic transcription, and NEFS is not trying to replace it for that purpose. The problems NEFS addresses are computational:

- **Scattered Unicode codepoints.** IPA symbols live across dozens of Unicode blocks. Comparing, sorting, or searching IPA strings requires traversing variable-length UTF-8 sequences with no structural relationship between similar sounds.
- **No encoded feature information.** In IPA, `p` and `b` look nothing alike, even though they differ by exactly one feature (voicing). A machine has no way to know this without a lookup table.
- **Poor parallelism.** Because IPA strings are variable-length and have no fixed structure, bulk phonetic operations cannot easily be vectorized.

NEFS encodes each phoneme as a **single byte**, with the byte value itself encoding phonological features. Similar sounds have numerically similar encodings. The result is a representation that fits in a fixed-width array, supports O(1) feature extraction, and can be processed in parallel across thousands of phonemes at once.

---

## The Encoding

NEFS uses a structured byte grid. Each byte encodes a phoneme, and the position of that byte in the grid encodes its features:

- **Rows** encode manner of articulation (stops, fricatives, nasals, approximants, clicks, etc.)
- **Columns** encode place of articulation (bilabial, labiodental, dental, alveolar, palatal, velar, uvular, glottal, etc.)
- **Diacritics and modifiers** occupy dedicated byte ranges and can be applied as independent bytes in a sequence, similar to Unicode combining characters — but with explicit positional semantics

Affricates are encoded as two-byte sequences where each byte is the corresponding stop and fricative component. For example, `tʃ` encodes as `[0x43, 0x34]` — `t` followed by `ʃ` — making the phonological structure of the affricate explicit in the encoding itself.

Tones are assigned a dedicated byte range (`0x04`–`0x0C`) covering level, rising, falling, and contour tones, with systematic byte spacing that reflects tonal relationships.

> **See the visual spec** (`spec/nefs-grid.png`) for the full byte grid and feature layout.
> **See the font file** (`spec/nefs.ttf`) for the NEFS script glyphs, which are also featurally organized — the visual shape of each symbol reflects its phonological features.

---

## The Two-Operation Guarantee

NEFS makes a claim that no prior phonetic encoding system makes:

> **Every phonological feature of every sound in the NEFS inventory is extractable by at most two logical operations on the encoding byte, with no auxiliary data structures required.**

This is a direct consequence of the byte grid design and is worth understanding concretely.

### What "two operations" means

In IPA, determining that `p` and `b` differ only in voicing requires a lookup table — the symbols have no algebraic relationship. In NEFS, voicing is a single bit in the low nibble. Extracting it is one AND and one compare. Determining that two bytes represent a minimal pair differing only in voicing is one XOR.

Every phonological feature works this way:

| Operation | IPA cost | NEFS cost |
|---|---|---|
| Extract place of articulation | lookup table | `B >> 4` |
| Extract manner of articulation | lookup table | `B & 0x0F` |
| Extract voicing (stops) | lookup table | `B & 0x01` |
| Extract modification class (plain/aspirated/NR) | lookup table | `(B >> 1) & 0x03` |
| Extract vowel height | lookup table | `B & 0x0F` |
| Test table membership (vowel/consonant/tone/etc.) | lookup table | at most two bitmask checks |
| Minimal pair test (one feature difference) | two lookups + compare | `popcount(A ^ B) == 1` |

### Why the vowel range is 0x_4–0x_B

This range was chosen specifically because it is the set of low nibble values where bits 3 and 2 differ — testable with a single XOR:

```c
int is_vowel = ((B >> 4) >= 0xA) && (((B >> 3) ^ (B >> 2)) & 1);
```

No range check on the low nibble is needed. The full table membership classification for all six NEFS regions requires at most two operations per region, with no lookup table at any point.

### What this enables

**Vectorized phonological search.** Any query over a phoneme corpus — "find all voiced stops," "find all front rounded vowels," "find all aspirated consonants" — compiles to at most two SIMD operations over the byte stream. On a CPU with 256-bit AVX registers that is 256 phonemes processed simultaneously per instruction pair. IPA requires a hash table lookup per symbol.

**Phonological distance.** Computing phonetic similarity between two sounds — how many features separate them — is `popcount(A ^ B)`. No tables. This makes large-scale rhyme detection, dialect comparison, and speech error analysis fast enough to run in real time over large corpora.

**Structured embeddings.** Neural TTS and ASR systems currently learn phoneme embeddings from scratch because IPA has no algebraic structure to exploit. NEFS byte values can seed phoneme embeddings directly from their feature structure, reducing training data requirements and improving generalization to low-resource languages.

**Hardware.** A phoneme processing unit targeting NEFS can implement full feature extraction in two gate delays. For embedded and microcontroller targets — including [SEER](../SEER), a microcontroller ISA developed alongside NEFS — this means speech processing extensions can be implemented without dedicated lookup ROM.

### Scope of the guarantee

The two-operation guarantee applies to the final grid layout after the planned bitfield reorganization. The current grid is partially consistent with this property; until the reorganization is complete, implementations should use the lookup table in `NEFSConverter` rather than relying on bitfield arithmetic directly.

---

## Relationship to IPA

NEFS is designed to be **losslessly interconvertible with IPA** for all sounds in the NEFS inventory. The `NEFSConverter` class provides:

```python
from nefs import NEFSConverter

converter = NEFSConverter()

# IPA → NEFS
nefs_bytes = converter.ipa_to_nefs("tʃ")   # → b'\x43\x34'

# NEFS → IPA
ipa_text = converter.nefs_to_ipa(b'\x43\x34')  # → 'tʃ'

# Lossless round-trip verification
assert converter.is_lossless("tʃ")  # True
```

Batch conversion is also supported for bulk processing:

```python
results = converter.batch_convert(["tʃ", "dʒ", "ts"], direction='ipa_to_nefs')
```

---

## SSML Integration

NEFS phoneme tags can be embedded in SSML for use with text-to-speech engines. The `NEFSTTSWrapper` handles conversion from NEFS-tagged SSML to IPA-tagged SSML before passing to downstream TTS providers (Amazon Polly, Azure TTS, Google Cloud TTS).

```python
from nefs.tts import create_nefs_adapter, AudioFormat, NEFSQuality, NEFSSynthesisRequest

tts = create_nefs_adapter('polly', api_key='...')

# Auto-generate SSML with NEFS phoneme tags
ssml = tts.create_nefs_ssml("Hello world")

# Synthesize
response = await tts.synthesize(NEFSSynthesisRequest(
    text=ssml,
    voice="neural-emma",
    language="en-US",
    format=AudioFormat.MP3,
    quality=NEFSQuality.NEURAL
))
```

> **Note:** The TTS wrapper is an integration layer. The core of this project is the encoding system and converter. The `_process_synthesis` method is currently stubbed for testing; real provider API calls require valid credentials and are implemented per adapter.

---

## Coverage

```python
converter = NEFSConverter()
stats = converter.get_stats()
# {
#   'total_ipa_mappings': ...,
#   'total_nafs_mappings': ...,
#   'affricate_mappings': 8,
#   'hex_coverage_percent': ...
# }
```

The encoding currently covers the full IPA consonant chart, vowel chart, tones, and common diacritics. Coverage is ongoing — see [open issues](#) for planned additions.

---

## Project Structure

```
nefs/
├── nefs/
│   ├── converter.py       # NEFSConverter — core IPA↔NEFS encoding
│   ├── tts/
│   │   ├── wrapper.py     # NEFSTTSWrapper — SSML processing and TTS integration
│   │   └── adapters.py    # Provider adapters (Polly, Azure, Google)
│   └── ssml.py            # SSMLNEFSProcessor — SSML parsing and NEFS tag handling
├── spec/
│   ├── nefs-grid.png      # Visual byte grid and feature layout  [coming soon]
│   └── nefs.ttf           # NEFS script font                     [coming soon]
├── tests/
└── README.md
```

---

## Status

NEFS is an active work in progress.

| Component | Status |
|---|---|
| Core encoding grid | ✅ Complete |
| IPA ↔ NEFS converter | ✅ Complete |
| Lossless round-trip verification | ✅ Complete |
| Affricate encoding | ✅ Complete |
| Tone encoding | ✅ Complete |
| SSML integration | ✅ Complete |
| TTS provider adapters | 🔧 Stubbed (integration pending) |
| Visual spec / font | 🔧 In progress |
| G2P integration (text → IPA → NEFS) | 📋 Planned |
| Full diacritic coverage | 📋 Planned |

---

## Contributing

Contributions are welcome, especially from those with backgrounds in linguistics, phonetics, or systems programming. Before contributing, please read the visual spec (`spec/nefs-grid.png`) to understand the design principles behind the byte grid — PRs that add phonemes should follow the existing row/column layout logic rather than appending arbitrarily.

Issues and discussion are open for:
- Sounds not yet in the inventory
- Edge cases in IPA conversion
- Performance benchmarks
- The formal spec (currently the code and visual spec together constitute the spec)

---

## License

[To be added]
