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

## Relationship to IPA

NEFS is designed to be **losslessly interconvertible with IPA** for all sounds in the NEFS inventory. The `NAFSConverter` class provides:

```python
from nefs import NAFSConverter

converter = NAFSConverter()

# IPA → NEFS
nefs_bytes = converter.ipa_to_nafs("tʃ")   # → b'\x43\x34'

# NEFS → IPA
ipa_text = converter.nafs_to_ipa(b'\x43\x34')  # → 'tʃ'

# Lossless round-trip verification
assert converter.is_lossless("tʃ")  # True
```

Batch conversion is also supported for bulk processing:

```python
results = converter.batch_convert(["tʃ", "dʒ", "ts"], direction='ipa_to_nafs')
```

---

## SSML Integration

NEFS phoneme tags can be embedded in SSML for use with text-to-speech engines. The `NAFSTTSWrapper` handles conversion from NEFS-tagged SSML to IPA-tagged SSML before passing to downstream TTS providers (Amazon Polly, Azure TTS, Google Cloud TTS).

```python
from nefs.tts import create_nefs_adapter, AudioFormat, NEFSQuality, NEFSSynthesisRequest

tts = create_nefs_adapter('polly', api_key='...')

# Auto-generate SSML with NEFS phoneme tags
ssml = tts.create_nafs_ssml("Hello world")

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
converter = NAFSConverter()
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
│   ├── converter.py       # NAFSConverter — core IPA↔NEFS encoding
│   ├── tts/
│   │   ├── wrapper.py     # NAFSTTSWrapper — SSML processing and TTS integration
│   │   └── adapters.py    # Provider adapters (Polly, Azure, Google)
│   └── ssml.py            # SSMLNEFSProcessor — SSML parsing and NAFS tag handling
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
