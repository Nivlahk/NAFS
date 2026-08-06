"""
nefs_converter.py -- NEFS <-> IPA bidirectional converter.
Derived directly from NEFS Specification v3, Appendix A.
byte = (high_nibble << 4) | low_nibble
high nibble = place column, low nibble = manner row.
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Tuple

_NEFS_TO_IPA: Dict[int, str] = {
    # Diacritic column 0x0_
    0x00:"\u02C8", 0x01:"\u0318", 0x02:"\u02CC", 0x03:"\u0319",
    0x04:"\u031D", 0x05:"\u031E", 0x06:"\u031F", 0x07:"\u0320",
    0x08:"\u02DE", 0x09:"\u0308", 0x0A:"\u0324", 0x0B:"\u0330",
    0x0C:"\u032C", 0x0D:"\u0325", 0x0E:"\u031A", 0x0F:"",
    # Bilabial 0x1_
    0x10:"\u03B2", 0x11:"\u0278", 0x12:"b",       0x13:"p",
    0x14:"b\u02B0",0x15:"p\u02B0",0x16:"b\u031A", 0x17:"p\u031A",
    0x18:"m",      0x19:"w",      0x1A:"\u028D",  0x1B:"\u0298",
    0x1C:"\u0253", 0x1D:"\u0299", 0x1E:"\u02B7",  0x1F:"",
    # Labiodental 0x2_
    0x20:"v",      0x21:"f",
    0x28:"\u0271", 0x29:"\u028B", 0x2A:"\u0265",
    0x2D:"\u2C71", 0x2E:"\u033C", 0x2F:".",
    # Dental 0x3_
    0x30:"\u00F0", 0x31:"\u03B8",
    0x3A:"\u026C", 0x3B:"\u01C0", 0x3C:"\u0257",
    0x3D:"\u027E", 0x3E:"\u032A", 0x3F:"\u0306",
    # Alveolar 0x4_
    0x40:"z",      0x41:"s",      0x42:"d",       0x43:"t",
    0x44:"d\u02B0",0x45:"t\u02B0",0x46:"d\u031A", 0x47:"t\u031A",
    0x48:"n",      0x49:"\u0279", 0x4A:"l",       0x4B:"\u01C1",
    0x4C:"\u0255", 0x4D:"r",      0x4E:"\u033A",  0x4F:"\u02D0",
    # Post-alveolar 0x5_
    0x50:"\u0292", 0x51:"\u0283",
    0x5A:"\u026E", 0x5B:"\u01C3", 0x5C:"\u0291",
    0x5D:"\u027A", 0x5E:"\u0303", 0x5F:"\u02E5",
    # Retroflex 0x6_
    0x60:"\u0290", 0x61:"\u0282",
    0x62:"\u0256", 0x63:"\u0288",
    0x64:"\u0256\u02B0", 0x65:"\u0288\u02B0",
    0x66:"\u0256\u031A", 0x67:"\u0288\u031A",
    0x68:"\u0273", 0x69:"\u027B", 0x6A:"\u026D",
    0x6D:"\u027D", 0x6E:"j\u02B2j\u02B2", 0x6F:"\u02E6",
    # Palatal 0x7_
    0x70:"\u029D", 0x71:"\u00E7",
    0x72:"\u025F", 0x73:"c",
    0x74:"\u025F\u02B0", 0x75:"c\u02B0",
    0x76:"\u025F\u031A", 0x77:"c\u031A",
    0x78:"\u0272", 0x79:"j",      0x7A:"\u028E",  0x7B:"\u01C2",
    0x7C:"\u0284", 0x7D:"\u0267", 0x7E:"j\u02B2", 0x7F:"\u02E7",
    # Velar 0x8_
    0x80:"\u0263", 0x81:"x",      0x82:"g",       0x83:"k",
    0x84:"g\u02B0",0x85:"k\u02B0",0x86:"g\u031A", 0x87:"k\u031A",
    0x88:"\u014B", 0x89:"\u0270", 0x8A:"\u029F",
    0x8C:"\u0260", 0x8E:"\u0263\u02E4", 0x8F:"\u02E8",
    # Uvular 0x9_
    0x90:"\u0281", 0x91:"\u03C7",
    0x92:"\u0262", 0x93:"q",
    0x94:"\u0262\u02B0", 0x95:"q\u02B0",
    0x96:"\u0262\u031A", 0x97:"q\u031A",
    0x98:"\u0274",
    0x9C:"\u029B", 0x9D:"\u0280",
    0x9E:"\u02E4\u02E4", 0x9F:"\u02E9",
    # Pharyngeal 0xA_
    0xA0:"\u0295", 0xA1:"\u0127",
    0xA4:"i",      0xA5:"e",      0xA6:"\u025B",  0xA7:"a",
    0xA8:"\u026A",
    0xA9:"0",      0xAA:"A",      0xAB:"D\u266F",
    0xAD:"Mk",     0xAE:"\u02E4",
    0xAF:"\u02E9\u02E5",
    # Epiglottal 0xB_
    0xB0:"\u02A1", 0xB1:"\u02A2",
    0xB4:"y",      0xB5:"\u00F8", 0xB6:"\u0153",  0xB7:"\u0276",
    0xB8:"\u028F",
    0xB9:"1",      0xBA:"A\u266F",0xBB:"E",
    0xBD:"Au",     0xBE:"\u02BC",
    0xBF:"\u02E5\u02E9",
    # Glottal 0xC_
    0xC0:"\u0266", 0xC1:"h",      0xC2:"\u0294",
    0xC4:"\u0268", 0xC5:"\u0258", 0xC6:"\u025C",  0xC7:"\u00E4",
    0xC8:"\u00E6",
    0xC9:"2",      0xCA:"B",      0xCB:"F",
    0xCD:"Vc",     0xCE:"\u02C0",
    0xCF:"\u02E6\u02E5",
    # Column 0xD_
    0xD4:"\u0289", 0xD5:"\u0275", 0xD6:"\u025E",  0xD7:"\u0252\u0308",
    0xD8:"\u0250",
    0xD9:"3",      0xDA:"C",      0xDB:"F\u266F",
    0xDD:"Ef",     0xDE:"h\u02B0",
    0xDF:"\u02E9\u02E8",
    # Column 0xE_
    0xE4:"\u026F", 0xE5:"\u0264", 0xE6:"\u028C",  0xE7:"\u0251",
    0xE8:"\u028A",
    0xE9:"4",      0xEA:"C\u266F",0xEB:"G",
    0xEE:"h\u02B0h\u02B0",
    0xEF:"\u02E7\u02E6\u02E7",
    # Column 0xF_
    0xF0:"\u2197", 0xF1:"\u2016", 0xF2:"|",
    0xF4:"u",      0xF5:"o",      0xF6:"\u0254",  0xF7:"\u0252",
    0xF8:"\u0259",
    0xF9:"5",      0xFA:"D",      0xFB:"G\u266F",
    0xFE:"\u033B", 0xFF:"\u2198",
}

_IPA_TO_NEFS: Dict[str, int] = {
    ipa: b for b, ipa in _NEFS_TO_IPA.items() if ipa
}
# eSpeak-specific aliases — normalise variant Unicode to spec bytes
_IPA_TO_NEFS['ɡ'] = 0x82   # U+0261 script g -> velar voiced stop
_IPA_TO_NEFS['ɧ'] = 0x7D   # U+0267 already in map, ensure present

_AFFRICATES: Dict[str, List[int]] = {
    "t\u0283":[0x43,0x51], "d\u0292":[0x42,0x50],
    "ts":      [0x43,0x41], "dz":      [0x42,0x40],
    "t\u0255":[0x43,0x4C], "d\u0291":[0x42,0x5C],
    "t\u0282":[0x43,0x61], "d\u0290":[0x42,0x60],
}
_AFFRICATE_PAIRS: Dict[Tuple[int,int], str] = {
    (v[0],v[1]): k for k,v in _AFFRICATES.items()
}


def classify_byte(b: int) -> str:
    if b == 0x0F: return "silence"
    low, high = b & 0x0F, b >> 4
    if low == 0x0F: return "tone"
    if low == 0x0E or high == 0x0: return "diacritic"
    if high >= 0xE and low <= 0x3: return "stress"
    if high >= 0xA and 0xC <= low <= 0xD: return "effects"
    if high >= 0xA and 0x4 <= low <= 0xB: return "vowel"
    return "consonant"


def is_voiced(b: int) -> bool:
    region = classify_byte(b)
    if region == "vowel": return True
    if region != "consonant": return False
    low = b & 0x0F
    if low == 0x0: return True
    if low == 0x1: return False
    if low in (0x2, 0x4, 0x6): return True
    if low in (0x3, 0x5, 0x7): return False
    return True  # nasals, approx, laterals, trills, implosives


def phoneme_class(b: int) -> str:
    region = classify_byte(b)
    if region == "vowel": return "vowel"
    if region != "consonant": return region
    low = b & 0x0F
    if low in (0x0, 0x1): return "fricative"
    if low in (0x2, 0x4, 0x6): return "stop"
    if low in (0x3, 0x5, 0x7): return "stop"
    if low == 0x8: return "nasal"
    if low == 0x9: return "approximant"
    if low == 0xA: return "lateral"
    if low == 0xB: return "click"
    if low == 0xC: return "implosive"
    if low == 0xD: return "trill"
    return "consonant"


def place(b: int) -> int: return (b >> 4) & 0x0F
def manner(b: int) -> int: return b & 0x0F


class NEFSConverter:
    def ipa_to_nefs(self, ipa: str) -> bytes:
        result: List[int] = []
        i = 0
        while i < len(ipa):
            matched = False
            for length in (3, 2):
                if i + length <= len(ipa):
                    sub = ipa[i:i+length]
                    if sub in _AFFRICATES:
                        result.extend(_AFFRICATES[sub]); i += length; matched = True; break
                    if sub in _IPA_TO_NEFS:
                        result.append(_IPA_TO_NEFS[sub]); i += length; matched = True; break
            if not matched:
                ch = ipa[i]
                if ch in _IPA_TO_NEFS: result.append(_IPA_TO_NEFS[ch])
                elif ch in _AFFRICATES: result.extend(_AFFRICATES[ch])
                else: warnings.warn(f"NEFSConverter: unmapped IPA '{ch}' (U+{ord(ch):04X}) skipped")
                i += 1
        return bytes(result)

    def nefs_to_ipa(self, nefs: bytes) -> str:
        result: List[str] = []
        i = 0
        while i < len(nefs):
            if i + 1 < len(nefs):
                pair = (nefs[i], nefs[i+1])
                if pair in _AFFRICATE_PAIRS:
                    result.append(_AFFRICATE_PAIRS[pair]); i += 2; continue
            b = nefs[i]
            ipa = _NEFS_TO_IPA.get(b)
            if ipa is not None: result.append(ipa)
            else: warnings.warn(f"NEFSConverter: unmapped byte 0x{b:02X} skipped")
            i += 1
        return "".join(result)

    def is_lossless(self, ipa: str) -> bool:
        try: return self.nefs_to_ipa(self.ipa_to_nefs(ipa)) == ipa
        except: return False

    def stats(self) -> dict:
        return {
            "nefs_to_ipa": len(_NEFS_TO_IPA),
            "ipa_to_nefs": len(_IPA_TO_NEFS),
            "affricates": len(_AFFRICATES),
            "coverage_pct": round(len(_NEFS_TO_IPA)/256*100, 1),
        }


__all__ = [
    "NEFSConverter", "classify_byte", "is_voiced", "phoneme_class",
    "place", "manner", "_NEFS_TO_IPA", "_IPA_TO_NEFS", "_AFFRICATES",
]
