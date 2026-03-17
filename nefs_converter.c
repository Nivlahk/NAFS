/*
 * nefs_converter.c
 *
 * Bidirectional IPA <-> NEFS encoder/decoder.
 *
 * The byte table here is the canonical NEFS encoding grid.  It matches
 * the Python NAFSConverter exactly.  Each byte encodes a single phoneme;
 * the byte value encodes phonological features:
 *
 *   Rows (high nibble)  — broad phonological class
 *   Columns (low nibble) — place / height within that class
 *
 * Aspirated stops occupy contiguous rows immediately above their plain
 * counterparts; unreleased stops occupy the row below.  This layout
 * makes feature comparison a matter of simple arithmetic.
 *
 * Compilation:
 *   gcc -O2 -Wall -Wextra -o nefs_converter nefs_converter.c
 *
 * Usage:
 *   ./nefs_converter                  (runs built-in round-trip test)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ─── Types ──────────────────────────────────────────────────────────────── */

typedef struct { const char *ipa; uint8_t nafs; } IPAToNAFS;
typedef struct { uint8_t nafs;    const char *ipa; } NAFSToIPA;

/* ─── IPA → NEFS table (canonical, matches Python NAFSConverter) ─────────
 *
 * Longer multi-character IPA sequences must appear BEFORE their component
 * characters so that greedy matching finds them first.  The converter uses
 * a hash-based trie for O(1) lookup; this table is just the initializer.
 *
 * Notation used for ASCII-safe representations of combining sequences:
 *   bʰ / dʰ etc.  — aspirated stops   (U+02B0 MODIFIER LETTER SMALL H)
 *   b̚  / d̚  etc.  — unreleased stops  (U+031A COMBINING LEFT ANGLE ABOVE)
 *   ^w^            — labialization modifier
 *   ^j^            — palatalization modifier
 *   ^jj^           — strong palatalization
 *   ^jjj^          — extra-strong palatalization
 *   ^h^            — breathy voice (single)
 *   ^hh^           — breathy voice (double)
 *   ^ɣ^            — velarization modifier
 *   ^ˀ^            — glottalization modifier
 */
static const IPAToNAFS ipa_to_nafs_table[] = {
    /* ── Multi-character sequences first (greedy matching requirement) ── */

    /* Tone contours */
    {"˧˦˧",   0x0D},   /* rising-falling contour */
    {"˩˥",    0x09},   /* low-rising contour      */
    {"˥˩",    0x0A},   /* high-falling contour    */
    {"˦˥",    0x0B},   /* mid-high falling        */
    {"˩˨",    0x0C},   /* low-mid falling         */

    /* Aspirated voiced stops */
    {"bʰ",    0x50},
    {"dʰ",    0x53},
    {"ɖʰ",    0x55},
    {"ɟʰ",    0x56},
    {"gʰ",    0x57},
    {"ɢʰ",    0x58},

    /* Aspirated voiceless stops */
    {"pʰ",    0x60},
    {"tʰ",    0x63},
    {"ʈʰ",    0x65},
    {"kʰ",    0x67},
    {"qʰ",    0x68},

    /* Unreleased voiced stops */
    {"b̚",     0x70},
    {"d̚",     0x73},
    {"ɖ̚",     0x75},
    {"ɟ̚",     0x76},
    {"g̚",     0x77},
    {"ɢ̚",     0x78},

    /* Unreleased voiceless stops */
    {"p̚",     0x80},
    {"t̚",     0x83},
    {"ʈ̚",     0x85},
    {"k̚",     0x87},
    {"q̚",     0x88},

    /* Rhotacized schwa */
    {"ə˞",    0x9F},   /* must precede bare ˞ */

    /* Pharyngealization double */
    {"ˤˤ",    0xF8},   /* must precede bare ˤ */

    /* Modifier sequences */
    {"^hh^",  0x2C},
    {"^h^",   0x1C},
    {"^w^",   0xF0},
    {"^jjj^", 0xF4},
    {"^jj^",  0xF5},
    {"^j^",   0xF6},
    {"^ɣ^",   0xF7},
    {"^ˀ^",   0xFB},

    /* ── Prosody & suprasegmentals ──────────────────────────────────── */
    {".",     0x01},   /* syllable break         */
    {"\u0306",0x02},   /* ̆  extra-short          */
    {"\u02D0",0x03},   /* ː  long                 */
    {"˥",     0x04},   /* extra-high tone         */
    {"˦",     0x05},   /* high tone               */
    {"˧",     0x06},   /* mid tone                */
    {"˨",     0x07},   /* low tone                */
    {"˩",     0x08},   /* extra-low tone          */
    {"ˈ",     0x0E},   /* primary stress          */
    {"ˌ",     0x0F},   /* secondary stress        */
    {"↘",     0xAE},   /* downstep                */
    {"↗",     0xBE},   /* upstep                  */

    /* ── Voiced fricatives (0x10 row) ──────────────────────────────── */
    {"β",     0x10},
    {"v",     0x11},
    {"ð",     0x12},
    {"z",     0x13},
    {"ʒ",     0x14},
    {"ʐ",     0x15},
    {"ʝ",     0x16},
    {"ɣ",     0x17},
    {"ʁ",     0x18},
    {"ʕ",     0x19},
    {"ʡ",     0x1A},
    {"ɦ",     0x1B},

    /* ── Diacritics (0x1D, 0x1F, miscellaneous) ────────────────────── */
    {"\u0324",0x1D},   /* ̤  breathy-voiced diacritic */
    {"˞",     0x1F},   /* rhotacization modifier      */

    /* ── Voiced stops (0x20 row) ────────────────────────────────────── */
    {"b",     0x20},
    {"d",     0x23},
    {"ɖ",     0x25},
    {"ɟ",     0x26},
    {"g",     0x27},
    {"\u0261",0x27},   /* ɡ  U+0261 — IPA-specific voiced velar stop glyph,
                        *    alias for ASCII g in NEFS (same phoneme, same byte) */
    {"ɢ",     0x28},
    {"ʔ",     0x2B},   /* glottal stop             */

    /* ── Diacritics (between stop rows) ────────────────────────────── */
    {"\u0330",0x2D},   /* ̰  creaky voice diacritic  */
    {"\u0318",0x2F},   /* ̘  ATR diacritic            */

    /* ── Voiceless fricatives (0x30 row) ───────────────────────────── */
    {"ɸ",     0x30},
    {"f",     0x31},
    {"θ",     0x32},
    {"s",     0x33},
    {"ʃ",     0x34},
    {"ʂ",     0x35},
    {"ç",     0x36},
    {"x",     0x37},
    {"χ",     0x38},
    {"ħ",     0x39},
    {"ʢ",     0x3A},
    {"h",     0x3B},

    /* ── Diacritics (between fricative and stop rows) ───────────────── */
    {"\u0325",0x3C},   /* ̥  voiceless diacritic      */
    {"\u032C",0x3D},   /* ̬  voiced diacritic          */
    {"\u033B",0x3E},   /* ̻  laminal diacritic         */
    {"\u0319",0x3F},   /* ̙  RTR diacritic             */

    /* ── Voiceless stops (0x40 row) ────────────────────────────────── */
    {"p",     0x40},
    {"t",     0x43},
    {"ʈ",     0x45},
    {"k",     0x47},
    {"q",     0x48},

    /* ── Close vowels (0x49–0x4E) ───────────────────────────────────── */
    {"i",     0x49},
    {"y",     0x4A},
    {"ɨ",     0x4B},
    {"ʉ",     0x4C},
    {"ɯ",     0x4D},
    {"u",     0x4E},
    {"\u031D",0x4F},   /* ̝  raised diacritic         */

    /* ── Close-mid vowels (0x59–0x5E) ──────────────────────────────── */
    {"e",     0x59},
    {"ø",     0x5A},
    {"ɘ",     0x5B},
    {"ɵ",     0x5C},
    {"ɤ",     0x5D},
    {"o",     0x5E},
    {"\u031E",0x5F},   /* ̞  lowered diacritic        */

    /* ── Open-mid vowels (0x69–0x6E) ───────────────────────────────── */
    {"ɛ",     0x69},
    {"œ",     0x6A},
    {"ɜ",     0x6B},
    {"ɞ",     0x6C},
    {"ʌ",     0x6D},
    {"ɔ",     0x6E},
    {"\u031F",0x6F},   /* ̟  advanced diacritic       */

    /* ── Open vowels (0x79–0x7E) ────────────────────────────────────── */
    {"a",     0x79},
    {"ɶ",     0x7A},
    {"ä",     0x7B},
    {"\u0252\u0308", 0x7C},  /* ɒ̈ */
    {"ɑ",     0x7D},
    {"ɒ",     0x7E},

    /* ── Near-close / near-open vowels (0x89–0x8E) ─────────────────── */
    {"ɪ",     0x89},
    {"ʏ",     0x8A},
    {"æ",     0x8B},
    {"ɐ",     0x8C},
    {"ʊ",     0x8D},
    {"ə",     0x8E},
    {"˞",     0x8F},   /* rhotacization (bare, after ə˞ multi-char) */

    /* ── Nasals (0x90 row) ──────────────────────────────────────────── */
    {"m",     0x90},
    {"ɱ",     0x91},
    {"n",     0x93},
    {"ɳ",     0x95},
    {"ɲ",     0x96},
    {"ŋ",     0x97},
    {"ɴ",     0x98},
    {"\u0303",0x9A},   /* ̃  nasalization diacritic    */
    {"\u0308",0x9F},   /* ̈  centralized diacritic     */

    /* ── Approximants (0xA0 row) ────────────────────────────────────── */
    {"w",     0xA0},
    {"ʋ",     0xA1},
    {"ɹ",     0xA3},
    {"ɻ",     0xA5},
    {"j",     0xA6},
    {"ɰ",     0xA7},

    /* ── Lateral approximants / fricatives (0xB0 row) ──────────────── */
    {"ʍ",     0xB0},
    {"ɥ",     0xB1},
    {"ɬ",     0xB2},
    {"l",     0xB3},
    {"ɮ",     0xB4},
    {"ɭ",     0xB5},
    {"ʎ",     0xB6},
    {"ʟ",     0xB7},

    /* ── Clicks (0xC0 row) ──────────────────────────────────────────── */
    {"ʘ",     0xC0},
    {"ǀ",     0xC2},
    {"ǁ",     0xC3},
    {"ǃ",     0xC4},
    {"ǂ",     0xC6},
    {"|",     0xCF},   /* minor prosodic boundary    */
    {"‖",     0xDF},   /* major prosodic boundary    */

    /* ── Implosives & clicks (0xD0 row) ─────────────────────────────── */
    {"ɓ",     0xD0},
    {"ɗ",     0xD2},
    {"ɕ",     0xD3},
    {"ʑ",     0xD4},
    {"ʄ",     0xD6},
    {"ɠ",     0xD7},
    {"ʛ",     0xD8},

    /* ── Trills, taps, flaps (0xE0 row) ────────────────────────────── */
    {"ʙ",     0xE0},
    {"ⱱ",     0xE1},
    {"ɾ",     0xE2},
    {"r",     0xE3},
    {"ɺ",     0xE4},
    {"ɽ",     0xE5},
    {"ɧ",     0xE6},
    {"ʀ",     0xE8},

    /* ── Secondary articulation modifiers (0xF0 row) ───────────────── */
    /* Multi-char sequences already handled above; bare variants follow */
    {"\u033C",0xF1},   /* ̼  linguolabial diacritic   */
    {"\u032A",0xF2},   /* ̪  dental diacritic          */
    {"\u033A",0xF3},   /* ̺  apical diacritic           */
    {"ˤ",     0xF9},   /* pharyngealization (bare)    */
    {"ʼ",     0xFB},   /* ejective                    */
    {"\u02DE",0xFC},   /* ˞  rhotic hook               */
    {"\u02BC",0xFD},   /* ʼ  modifier apostrophe       */
    {"\u031A",0xFE},   /* ̚  unreleased diacritic (bare) */
};

/* ─── NEFS → IPA table (canonical reverse map) ──────────────────────────── */
static const NAFSToIPA nafs_to_ipa_table[] = {
    {0x01, "."},
    {0x02, "\u0306"},   /* ̆ */
    {0x03, "\u02D0"},   /* ː */
    {0x04, "˥"},
    {0x05, "˦"},
    {0x06, "˧"},
    {0x07, "˨"},
    {0x08, "˩"},
    {0x09, "˩˥"},
    {0x0A, "˥˩"},
    {0x0B, "˦˥"},
    {0x0C, "˩˨"},
    {0x0D, "˧˦˧"},
    {0x0E, "\u02C8"},   /* ˈ */
    {0x0F, "\u02CC"},   /* ˌ */
    {0x10, "\u03B2"},   /* β */
    {0x11, "v"},
    {0x12, "\u00F0"},   /* ð */
    {0x13, "z"},
    {0x14, "\u0292"},   /* ʒ */
    {0x15, "\u0290"},   /* ʐ */
    {0x16, "\u029D"},   /* ʝ */
    {0x17, "\u0263"},   /* ɣ */
    {0x18, "\u0281"},   /* ʁ */
    {0x19, "\u0295"},   /* ʕ */
    {0x1A, "\u02A1"},   /* ʡ */
    {0x1B, "\u0266"},   /* ɦ */
    {0x1C, "h\u02B0"},  /* hʰ — breathy-voiced h     */
    {0x1D, "\u0324"},   /* ̤  */
    {0x1F, "\u02DE"},   /* ˞  */
    {0x20, "b"},
    {0x23, "d"},
    {0x25, "\u0256"},   /* ɖ */
    {0x26, "\u025F"},   /* ɟ */
    {0x27, "g"},
    {0x28, "\u0262"},   /* ɢ */
    {0x2B, "\u0294"},   /* ʔ */
    {0x2C, "h\u02B0h\u02B0"}, /* hhʰ — double breathy */
    {0x2D, "\u0330"},   /* ̰  */
    {0x2F, "\u0318"},   /* ̘  */
    {0x30, "\u0278"},   /* ɸ */
    {0x31, "f"},
    {0x32, "\u03B8"},   /* θ */
    {0x33, "s"},
    {0x34, "\u0283"},   /* ʃ */
    {0x35, "\u0282"},   /* ʂ */
    {0x36, "\u00E7"},   /* ç */
    {0x37, "x"},
    {0x38, "\u03C7"},   /* χ */
    {0x39, "\u0127"},   /* ħ */
    {0x3A, "\u02A2"},   /* ʢ */
    {0x3B, "h"},
    {0x3C, "\u0325"},   /* ̥  */
    {0x3D, "\u032C"},   /* ̬  */
    {0x3E, "\u033B"},   /* ̻  */
    {0x3F, "\u0319"},   /* ̙  */
    {0x40, "p"},
    {0x43, "t"},
    {0x45, "\u0288"},   /* ʈ */
    {0x46, "\u025F"},   /* ɟ */
    {0x47, "k"},
    {0x48, "q"},
    {0x49, "i"},
    {0x4A, "y"},
    {0x4B, "\u0268"},   /* ɨ */
    {0x4C, "\u0289"},   /* ʉ */
    {0x4D, "\u026F"},   /* ɯ */
    {0x4E, "u"},
    {0x4F, "\u031D"},   /* ̝  */
    {0x50, "b\u02B0"},  /* bʰ */
    {0x53, "d\u02B0"},  /* dʰ */
    {0x55, "\u0256\u02B0"}, /* ɖʰ */
    {0x56, "\u025F\u02B0"}, /* ɟʰ */
    {0x57, "g\u02B0"},  /* gʰ */
    {0x58, "\u0262\u02B0"}, /* ɢʰ */
    {0x59, "e"},
    {0x5A, "\u00F8"},   /* ø */
    {0x5B, "\u0258"},   /* ɘ */
    {0x5C, "\u0275"},   /* ɵ */
    {0x5D, "\u0264"},   /* ɤ */
    {0x5E, "o"},
    {0x5F, "\u031E"},   /* ̞  */
    {0x60, "p\u02B0"},  /* pʰ */
    {0x63, "t\u02B0"},  /* tʰ */
    {0x65, "\u0288\u02B0"}, /* ʈʰ */
    {0x66, "\u025F\u02B0"}, /* ɟʰ (aspirated palatal) */
    {0x67, "k\u02B0"},  /* kʰ */
    {0x68, "q\u02B0"},  /* qʰ */
    {0x69, "\u025B"},   /* ɛ */
    {0x6A, "\u0153"},   /* œ */
    {0x6B, "\u025C"},   /* ɜ */
    {0x6C, "\u025E"},   /* ɞ */
    {0x6D, "\u028C"},   /* ʌ */
    {0x6E, "\u0254"},   /* ɔ */
    {0x6F, "\u031F"},   /* ̟  */
    {0x70, "b\u031A"},  /* b̚ */
    {0x73, "d\u031A"},  /* d̚ */
    {0x75, "\u0256\u031A"}, /* ɖ̚ */
    {0x76, "\u025F\u031A"}, /* ɟ̚ */
    {0x77, "g\u031A"},  /* g̚ */
    {0x78, "\u0262\u031A"}, /* ɢ̚ */
    {0x79, "a"},
    {0x7A, "\u0276"},   /* ɶ */
    {0x7B, "\u00E4"},   /* ä */
    {0x7C, "\u0252\u0308"}, /* ɒ̈ */
    {0x7D, "\u0251"},   /* ɑ */
    {0x7E, "\u0252"},   /* ɒ */
    {0x7F, "\u031F"},   /* ̟  (RTR variant) */
    {0x80, "p\u031A"},  /* p̚ */
    {0x83, "t\u031A"},  /* t̚ */
    {0x85, "\u0288\u031A"}, /* ʈ̚ */
    {0x86, "\u025F\u031A"}, /* ɟ̚ */
    {0x87, "k\u031A"},  /* k̚ */
    {0x88, "q\u031A"},  /* q̚ */
    {0x89, "\u026A"},   /* ɪ */
    {0x8A, "\u028F"},   /* ʏ */
    {0x8B, "\u00E6"},   /* æ */
    {0x8C, "\u0250"},   /* ɐ */
    {0x8D, "\u028A"},   /* ʊ */
    {0x8E, "\u0259"},   /* ə */
    {0x8F, "\u02DE"},   /* ˞  (rhotacization) */
    {0x90, "m"},
    {0x91, "\u0271"},   /* ɱ */
    {0x93, "n"},
    {0x95, "\u0273"},   /* ɳ */
    {0x96, "\u0272"},   /* ɲ */
    {0x97, "\u014B"},   /* ŋ */
    {0x98, "\u0274"},   /* ɴ */
    {0x99, "\u0303"},   /* ̃  nasalization */
    {0x9F, "\u0308"},   /* ̈  centralized */
    {0xA0, "w"},
    {0xA1, "\u028B"},   /* ʋ */
    {0xA3, "\u0279"},   /* ɹ */
    {0xA5, "\u027B"},   /* ɻ */
    {0xA6, "j"},
    {0xA7, "\u0270"},   /* ɰ */
    {0xAE, "\u2198"},   /* ↘ downstep */
    {0xB0, "\u028D"},   /* ʍ */
    {0xB1, "\u0265"},   /* ɥ */
    {0xB2, "\u026C"},   /* ɬ */
    {0xB3, "l"},
    {0xB4, "\u026E"},   /* ɮ */
    {0xB5, "\u026D"},   /* ɭ */
    {0xB6, "\u028E"},   /* ʎ */
    {0xB7, "\u029F"},   /* ʟ */
    {0xBE, "\u2197"},   /* ↗ upstep */
    {0xC0, "\u0298"},   /* ʘ bilabial click */
    {0xC2, "\u01C0"},   /* ǀ dental click */
    {0xC3, "\u01C1"},   /* ǁ lateral click */
    {0xC4, "\u01C3"},   /* ǃ alveolar click */
    {0xC6, "\u01C2"},   /* ǂ palatal click */
    {0xCF, "|"},        /* minor prosodic boundary */
    {0xD0, "\u0253"},   /* ɓ */
    {0xD2, "\u0257"},   /* ɗ */
    {0xD3, "\u0255"},   /* ɕ */
    {0xD4, "\u0291"},   /* ʑ */
    {0xD6, "\u0284"},   /* ʄ */
    {0xD7, "\u0260"},   /* ɠ */
    {0xD8, "\u029B"},   /* ʛ */
    {0xDF, "\u2016"},   /* ‖ major prosodic boundary */
    {0xE0, "\u0299"},   /* ʙ */
    {0xE1, "\u2C71"},   /* ⱱ */
    {0xE2, "\u027E"},   /* ɾ */
    {0xE3, "r"},
    {0xE4, "\u027A"},   /* ɺ */
    {0xE5, "\u027D"},   /* ɽ */
    {0xE6, "\u0267"},   /* ɧ */
    {0xE8, "\u0280"},   /* ʀ */
    {0xF0, "w\u02B7"},  /* ʷ labialization modifier  */
    {0xF1, "\u033C"},   /* ̼  linguolabial */
    {0xF2, "\u032A"},   /* ̪  dental */
    {0xF3, "\u033A"},   /* ̺  apical */
    {0xF4, "j\u02B2j\u02B2j\u02B2"}, /* ʲʲʲ extra-strong palatalization */
    {0xF5, "j\u02B2j\u02B2"},        /* ʲʲ  strong palatalization */
    {0xF6, "j\u02B2"},               /* ʲ   palatalization */
    {0xF7, "\u0263\u02E4"},          /* ɣˤ  velarization */
    {0xF8, "\u02E4\u02E4"},          /* ˤˤ  double pharyngealization */
    {0xF9, "\u02E4"},                /* ˤ   pharyngealization */
    {0xFB, "\u02C0"},   /* ˀ glottalization */
    {0xFC, "\u02DE"},   /* ˞  rhotic hook */
    {0xFD, "\u02BC"},   /* ʼ  modifier apostrophe */
    {0xFE, "\u031A"},   /* ̚   unreleased diacritic */
};

#define IPA_TO_NAFS_SIZE (sizeof(ipa_to_nafs_table) / sizeof(ipa_to_nafs_table[0]))
#define NAFS_TO_IPA_SIZE (sizeof(nafs_to_ipa_table) / sizeof(nafs_to_ipa_table[0]))

/* ─── UTF-8 utilities ────────────────────────────────────────────────────── */

/* Return the byte-length of the UTF-8 sequence starting at first_byte. */
static int utf8_char_length(unsigned char first_byte) {
    if ((first_byte & 0x80) == 0x00) return 1;
    if ((first_byte & 0xE0) == 0xC0) return 2;
    if ((first_byte & 0xF0) == 0xE0) return 3;
    if ((first_byte & 0xF8) == 0xF0) return 4;
    return 1; /* invalid byte — treat as single */
}

/*
 * Copy the UTF-8 character at str[pos] into char_buf (NUL-terminated).
 * Returns the byte length of the character, or 0 on error.
 */
static int utf8_extract_char(const char *str, int pos, int str_len,
                              char *char_buf, int buf_size) {
    if (pos >= str_len || buf_size < 2) return 0;
    int len = utf8_char_length((unsigned char)str[pos]);
    if (pos + len > str_len || len >= buf_size) return 0;
    memcpy(char_buf, str + pos, len);
    char_buf[len] = '\0';
    return len;
}

/* ─── Forward declaration for build_ipa_lookup ───────────────────────────── */

/*
 * Simple open-addressed hash table for IPA string → NAFS byte.
 * We use the table at startup and never mutate it, so there are
 * no concurrency concerns.
 */
#define HASH_SIZE 512   /* must be power of two, > 2 × IPA_TO_NAFS_SIZE */

typedef struct {
    const char *key;   /* NULL = empty slot */
    uint8_t     value;
} HashEntry;

static HashEntry ipa_hash[HASH_SIZE];
static int ipa_hash_built = 0;

static unsigned int hash_str(const char *s) {
    unsigned int h = 2166136261u;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

static void build_ipa_lookup(void) {
    if (ipa_hash_built) return;
    memset(ipa_hash, 0, sizeof(ipa_hash));
    for (size_t i = 0; i < IPA_TO_NAFS_SIZE; i++) {
        unsigned int slot = hash_str(ipa_to_nafs_table[i].ipa) & (HASH_SIZE - 1);
        while (ipa_hash[slot].key != NULL)
            slot = (slot + 1) & (HASH_SIZE - 1);
        ipa_hash[slot].key   = ipa_to_nafs_table[i].ipa;
        ipa_hash[slot].value = ipa_to_nafs_table[i].nafs;
    }
    ipa_hash_built = 1;
}

/*
 * Look up an IPA string in the hash table.
 * Returns the NAFS byte, or 0xFF (unused sentinel) on miss.
 */
static int ipa_lookup(const char *key, uint8_t *out) {
    unsigned int slot = hash_str(key) & (HASH_SIZE - 1);
    while (ipa_hash[slot].key != NULL) {
        if (strcmp(ipa_hash[slot].key, key) == 0) {
            *out = ipa_hash[slot].value;
            return 1;
        }
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
    return 0;
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

/*
 * ipa_to_nafs()
 *
 * Convert a NUL-terminated IPA string to a NEFS byte sequence.
 * Uses greedy longest-match via hash lookup; O(n) in input length.
 *
 * Parameters:
 *   ipa_string      — input IPA text (UTF-8)
 *   output          — caller-supplied output buffer
 *   max_output_size — capacity of output buffer
 *
 * Returns the number of NEFS bytes written.
 * Unrecognised IPA characters are silently skipped (emit a warning via
 * stderr so callers can detect coverage gaps during development).
 */
int ipa_to_nafs(const char *ipa_string, uint8_t *output, int max_output_size) {
    build_ipa_lookup();

    int output_len = 0;
    int pos        = 0;
    int input_len  = (int)strlen(ipa_string);

    /*
     * Greedy matching strategy:
     *   At each position we try the longest possible substring first.
     *   We build candidates by appending one UTF-8 character at a time,
     *   keeping the longest match seen so far.  When we can no longer
     *   extend the candidate (end of string or buffer full), we emit the
     *   best match found and advance past it.
     *
     *   Worst case: O(n × max_candidate_chars) where max_candidate_chars
     *   is the length of the longest key in the table (≈ 6 UTF-8 chars).
     *   For practical IPA strings this is effectively O(n).
     */
    while (pos < input_len && output_len < max_output_size) {
        char   candidate[32] = {0};  /* accumulated substring being tested */
        int    cand_bytes    = 0;    /* byte length of candidate            */
        int    cand_chars    = 0;    /* char count of candidate             */
        int    best_advance  = 0;    /* byte advance for best match         */
        uint8_t best_nafs    = 0;

        int scan = pos;

        /* Try appending characters one at a time, recording each match. */
        while (scan < input_len && cand_chars < 8) {
            char utf8_char[8] = {0};
            int char_len = utf8_extract_char(ipa_string, scan, input_len,
                                             utf8_char, sizeof(utf8_char));
            if (char_len == 0) { scan++; break; }
            if (cand_bytes + char_len >= (int)sizeof(candidate)) break;

            memcpy(candidate + cand_bytes, utf8_char, char_len);
            cand_bytes += char_len;
            candidate[cand_bytes] = '\0';
            cand_chars++;
            scan += char_len;

            uint8_t nafs_byte;
            if (ipa_lookup(candidate, &nafs_byte)) {
                best_nafs    = nafs_byte;
                best_advance = scan - pos;   /* remember the longest match */
            }
        }

        if (best_advance > 0) {
            output[output_len++] = best_nafs;
            pos += best_advance;
        } else {
            /* No match: skip this UTF-8 character */
            char skip[8] = {0};
            int skip_len = utf8_extract_char(ipa_string, pos, input_len,
                                             skip, sizeof(skip));
            if (skip_len == 0) skip_len = 1;
            fprintf(stderr, "nefs_converter: unmapped IPA at byte %d: ",  pos);
            for (int k = 0; k < skip_len; k++)
                fprintf(stderr, "%02X", (unsigned char)ipa_string[pos + k]);
            fprintf(stderr, "\n");
            pos += skip_len;
        }
    }

    return output_len;
}

/*
 * nafs_to_ipa()
 *
 * Convert a NEFS byte sequence to a NUL-terminated IPA string.
 * Uses a 256-entry direct-indexed lookup; O(n) in input length.
 *
 * Parameters:
 *   nafs_bytes      — input NEFS bytes
 *   nafs_len        — number of input bytes
 *   output          — caller-supplied output buffer (UTF-8)
 *   max_output_size — capacity of output buffer (including NUL)
 *
 * Returns the number of bytes written (excluding NUL).
 * Unrecognised NEFS bytes are silently skipped.
 */
int nafs_to_ipa(const uint8_t *nafs_bytes, int nafs_len,
                char *output, int max_output_size) {
    /* Build direct-index reverse table once. */
    static const char *reverse[256] = {NULL};
    static int reverse_built = 0;
    if (!reverse_built) {
        for (size_t i = 0; i < NAFS_TO_IPA_SIZE; i++)
            reverse[nafs_to_ipa_table[i].nafs] = nafs_to_ipa_table[i].ipa;
        reverse_built = 1;
    }

    int out = 0;
    for (int i = 0; i < nafs_len; i++) {
        const char *sym = reverse[nafs_bytes[i]];
        if (!sym) {
            fprintf(stderr, "nefs_converter: unmapped NEFS byte 0x%02X at position %d\n",
                    nafs_bytes[i], i);
            continue;
        }
        int sym_len = (int)strlen(sym);
        if (out + sym_len >= max_output_size) {
            fprintf(stderr, "nefs_converter: output buffer full at position %d\n", i);
            break;
        }
        memcpy(output + out, sym, sym_len);
        out += sym_len;
    }
    output[out] = '\0';
    return out;
}

/* ─── Self-test ──────────────────────────────────────────────────────────── */

int main(void) {
    /* Test strings: IPA transcriptions of "hello" and a phonologically
     * rich phrase exercising stops, fricatives, affricates, vowels, tones. */
    static const char *tests[] = {
        "həloʊ",          /* hello        */
        "wɜːld",          /* world        */
        "tʃɪldɹən",       /* children     */
        "˧m˨a˧",          /* ma (Mandarin mid-falling-mid) */
        "bʰaːg",     /* bʰaːg  (aspirated b, long a, velar stop) */
        "g̚læs",    /* g̚læs  (unreleased g, lateral, front vowel, s) */
        NULL
    };

    printf("NEFS round-trip test\n");
    printf("%-20s  %-30s  %s\n", "IPA input", "NEFS (hex)", "IPA reconstructed");
    printf("%-20s  %-30s  %s\n", "---------", "----------", "-----------------");

    int all_pass = 1;

    for (int t = 0; tests[t] != NULL; t++) {
        const char *ipa_in = tests[t];
        uint8_t nafs_buf[256];
        char    ipa_out[1024];

        int nafs_len = ipa_to_nafs(ipa_in, nafs_buf, (int)sizeof(nafs_buf));

        char hex_str[256] = {0};
        for (int i = 0; i < nafs_len; i++) {
            char tmp[8];
            snprintf(tmp, sizeof(tmp), "%02X", nafs_buf[i]);
            if (i) strcat(hex_str, " ");
            strcat(hex_str, tmp);
        }

        nafs_to_ipa(nafs_buf, nafs_len, ipa_out, (int)sizeof(ipa_out));

        int match = (strcmp(ipa_in, ipa_out) == 0);
        if (!match) all_pass = 0;

        printf("%-20s  %-30s  %s  %s\n",
               ipa_in, hex_str, ipa_out, match ? "✓" : "✗ MISMATCH");
    }

    printf("\n%s\n", all_pass ? "All tests passed." : "FAILURES detected — check stderr for unmapped symbols.");
    return all_pass ? 0 : 1;
}
