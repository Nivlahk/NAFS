# Phylph

Phonetics tooling built around a featural representation of speech sounds — a byte encoding for machines, and a script plus teaching site for humans.

The repo holds three things that share one underlying idea: that a sound's phonological features should be visible in how it's written, whether the reader is a CPU or a student.

| Part | What it is | State |
|---|---|---|
| **NEFS** | A byte encoding for phonemes. Feature structure lives in the byte value. | Working, 79% IPA coverage |
| **Tessera** | A featural script and font, plus a browser-based teaching site. | Working, in active use |
| **TTS stack** | Experimental neural and formant synthesis driven by NEFS. | Research code, unstable |

Licensed GPL-3.0.

---

## Quickstart

The encoder has no dependencies beyond the standard library.

```bash
git clone https://github.com/Nivlahk/Phylph
cd Phylph
python3
```

```python
from nefs_converter import NEFSConverter, place, manner, is_voiced, phoneme_class

c = NEFSConverter()

c.ipa_to_nefs("tʃ")            # b'\x43\x51'  — t, then ʃ
c.nefs_to_ipa(b'\x43\x51')     # 'tʃ'
c.is_lossless("tʃ")            # True

c.stats()
# {'nefs_to_ipa': 203, 'ipa_to_nefs': 202, 'affricates': 8, 'coverage_pct': 79.3}

place(0x43)                    # 4  — alveolar
manner(0x43)                   # 3
is_voiced(0x43)                # False  (0x42 is 'd')
phoneme_class(0x43)            # consonant
```

To browse the teaching site, serve the repo root and open `Phlyph/index.html`:

```bash
python3 -m http.server 8000
# then visit http://localhost:8000/Phlyph/
```

The TTS scripts need a much heavier environment — see [TTS stack](#tts-stack) below.

---

## NEFS — the encoding

Each phoneme is one byte. The byte's value encodes its features, so sounds that are phonologically similar are numerically similar.

- **High nibble** selects the place column (`0x1_` bilabial, `0x2_` labiodental, `0x3_` dental, `0x4_` alveolar, `0x5_` post-alveolar, `0x6_` retroflex, `0x7_` palatal, and so on). `0x0_` is reserved for diacritics.
- **Low nibble** selects the manner row within that column, with voicing distinguished in the low bit for stops and fricatives.

So `t` is `0x43` and `d` is `0x42` — a single bit apart, because voicing is a single bit. Feature comparison becomes arithmetic:

```python
bin(0x43 ^ 0x42).count("1") == 1   # t and d are a minimal pair
```

**Affricates** are two-byte sequences of their stop and fricative components, which keeps the phonological structure explicit: `tʃ` is `0x43 0x51`, not an opaque third symbol.

**Tones** occupy a dedicated byte range covering level, rising, falling, and contour tones.

Current inventory: 203 byte→IPA mappings, 202 in the reverse direction, 8 affricates, at 79.3% of the target grid. The gaps are mostly rarer diacritics and the sparser place columns.

### Why bother

IPA is the right tool for human transcription and NEFS does not try to replace it there. The problems NEFS targets are computational:

- IPA symbols are scattered across many Unicode blocks as variable-length UTF-8, so there is no structural relationship between similar sounds.
- Nothing in the encoding of `p` and `b` reveals that they differ by one feature. A machine needs a lookup table.
- Variable-length symbols make bulk phonological operations hard to vectorize.

With a fixed-width feature-bearing byte, queries like *find all voiced stops* become mask-and-compare over a byte array, and phonological distance between two sounds becomes a popcount of their XOR. That's the argument. It has not yet been benchmarked against a tuned IPA lookup implementation, and until it has, treat the speedup as a design rationale rather than a measured result.

### On the "two operation" property

An earlier version of this document claimed that every phonological feature is extractable in at most two logical operations with no auxiliary structures. That is the **design target for the grid, not a description of the current one**.

Place and manner do extract in one operation each (`b >> 4`, `b & 0x0F`), and voicing for stops is one mask. But the grid has grown unevenly, several regions are not yet bitfield-consistent, and table membership tests currently need more than two operations. A bitfield reorganization to close the gap is planned but not done.

Until then, use `NEFSConverter` and the helper functions in `nefs_converter.py` rather than doing bitfield arithmetic against the raw bytes yourself. The helpers will keep working across the reorganization; hand-rolled masks will not.

---

## Tessera — the script and teaching site

Tessera is a featural script for the IPA, aimed at English speakers learning new sounds. Glyph shape reflects phonological features, so a learner can see what changes between two sounds instead of memorizing unrelated symbols.

`Phlyph/` is a static site — plain HTML, CSS, and JavaScript, no build step:

| Page | What it does |
|---|---|
| `index.html` | Entry point and overview of the script |
| `consonants.html` | English consonant grid rendered in Tessera |
| `vowels.html` | English vowel grid rendered in Tessera |
| `mouthmap.html` | English consonants arranged by place of articulation |
| `lab.html` | Build a sound feature by feature and watch the glyph change |
| `slate.html` | Type in IPA, see it in Tessera |
| `vot-simulator.html` | Interactive voice onset timing |
| `ballon.html` | Manner of articulation as a fluid simulation |
| `mandarin-sounds.html` | Mandarin sounds pitched at English speakers |

`Phlyph/Place Images/` holds sagittal articulation diagrams for the ten major places. `Phlyph/Tessera.ttf` is the font the pages render with.

Additional fonts live in `NAFS(Font)/`, `HEXD/`, and `NIVLAC(Font)/`, each with a cheat sheet.

---

## TTS stack

Experimental, and the least finished part of the repo. NEFS is used as the phoneme representation feeding several synthesis paths.

| File | Role |
|---|---|
| `nefs_g2p.py` | Grapheme to phoneme, text → IPA → NEFS |
| `nefs_wrapper.py` | SSML processing and TTS provider adapters |
| `nefs_klatt.py` | Klatt formant synthesizer |
| `nefs_tts_hifigan.py`, `nefs_hifigan_discriminator.py` | Neural vocoder |
| `nefs_param_predictor.py`, `nefs_prosody_extractor.py` | Synthesis parameter and prosody models |
| `nefs_espeak_rt.py`, `nefs_espeak_bootstrap.py` | eSpeak backend and bootstrapping |
| `nefs_polly_backend.py`, `nefs_polly_postprocess.py` | Amazon Polly integration |
| `train_nefs_tts.py`, `train_nefs_predictor.py` | Training entry points, see `TRAINING_GUIDE.md` |

Requires `torch`, `torchaudio`, `transformers`, `librosa`, `crepe`, `scipy`, `numpy`, `soundfile`, `sounddevice`, `textgrid`, `tqdm`, plus `boto3` for Polly and `epitran`/`phonemizer` for G2P. There is no pinned requirements file yet.

Cloud provider adapters need credentials. The base `_process_synthesis` is a stub; each provider adapter overrides it.

---

## Also in the repo

- `nefs_converter.c` — C implementation of the encoding grid. Note that its header comment describes the nibble layout differently from the Python version; the Python module is authoritative until they're reconciled.
- `NEFS_Specification_v3.docx` — the fullest written spec.
- `NAFS_Complete_Guide.md` — script guide.
- `Syllabic bytes.png` — syllable-level byte layout.
- `nefs_demo.html` — standalone encoding demo.
- `nefs_testsuite.py` — TTS benchmarking harness. Standard library only, but it exercises the wrapper rather than the core converter.

---

## Naming

The project has accumulated several names and they are being consolidated. Current intent:

- **Phylph** — the project and repo
- **NEFS** — the byte encoding
- **Tessera** — the script, font, and teaching site

`NAFS` appears throughout the older code and documents as a prior name for NEFS. The two are the same system. The `Phlyph/` directory name is a leftover transposition of `Phylph` and will be renamed.

---

## Status

| Component | State |
|---|---|
| IPA ↔ NEFS converter | Working, lossless round-trip verified |
| Encoding grid | 79.3% coverage, bitfield reorganization pending |
| Affricate and tone encoding | Working |
| C implementation | Working, nibble docs need reconciling with Python |
| Tessera font and teaching site | Working, in active use |
| SSML processing | Working |
| TTS provider adapters | Credentials required, base synthesis stubbed |
| G2P pipeline | Experimental |
| Neural synthesis | Research code, unstable |
| Formal written spec | Only in `.docx`, needs a Markdown version |
| Benchmarks | None yet |
| Packaging and tests | None yet |

---

## Contributing

Most useful right now:

- **Filling the grid.** Missing phonemes should follow the existing place-column and manner-row logic rather than taking the next free byte. Read `NEFS_Specification_v3.docx` first.
- **Benchmarks.** The performance case for NEFS is currently theoretical. A comparison against a tuned IPA hash-table implementation would either support it or kill it, and both outcomes are useful.
- **Conversion edge cases.** Round-trip failures and IPA sequences that don't map cleanly.
- **Repo hygiene.** A `.gitignore`, a `requirements.txt`, a real test suite for the converter, and a proper Python package layout.

Open an issue before large structural changes so we don't collide on the grid reorganization.

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
