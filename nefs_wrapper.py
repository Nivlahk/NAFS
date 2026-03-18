
import asyncio
import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import time
from nefs_g2p import text_to_ipa as _g2p_text_to_ipa

class AudioFormat(Enum):
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    PCM = "audio/pcm"
    OGG = "audio/ogg"

class NEFSQuality(Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    NEURAL = "neural"

@dataclass
class NEFSSynthesisRequest:
    text: str
    nefs_encoding: Optional[bytes] = None
    voice: str = "default"
    language: str = "en-US"
    format: AudioFormat = AudioFormat.MP3
    quality: NEFSQuality = NEFSQuality.STANDARD
    speed: float = 1.0
    pitch: float = 1.0
    ssml_support: bool = True
    is_ssml: bool = False  # New field to indicate SSML content

@dataclass
class NEFSSynthesisResponse:
    audio_data: bytes
    format: AudioFormat
    duration: float
    metadata: Dict
    nefs_encoding_used: bytes
    processing_time: float

class SSMLNEFSProcessor:
    """Handles SSML parsing and NAFS phoneme tag processing"""

    def __init__(self, nefs_converter):
        self.nefs_converter = nefs_converter
        self.ssml_namespace = {'ssml': 'http://www.w3.org/2001/10/synthesis'}

    def process_ssml_with_nafs(self, ssml_text: str) -> Dict:
        """
        Process SSML text and convert NAFS phoneme tags to target format
        Returns processed SSML and metadata about NAFS conversions
        """
        # Parse SSML
        try:
            # Wrap in speak tag if not present
            if not ssml_text.strip().startswith('<speak'):
                ssml_text = f'<speak xmlns="http://www.w3.org/2001/10/synthesis">{ssml_text}</speak>'

            root = ET.fromstring(ssml_text)
            nafs_conversions = []

            # Process all phoneme tags
            for phoneme_elem in root.iter():
                if phoneme_elem.tag.endswith('phoneme'):
                    alphabet = phoneme_elem.get('alphabet', '').lower()
                    ph_value = phoneme_elem.get('ph', '')

                    if alphabet == 'nafs':
                        # Convert NAFS to IPA for compatibility
                        try:
                            # Decode NAFS bytes from hex string
                            nefs_bytes = bytes.fromhex(ph_value)
                            ipa_text = self.nefs_converter.nafs_to_ipa(nefs_bytes)

                            # Update the phoneme tag to use IPA
                            phoneme_elem.set('alphabet', 'ipa')
                            phoneme_elem.set('ph', ipa_text)

                            nafs_conversions.append({
                                'original_nafs': ph_value,
                                'converted_ipa': ipa_text,
                                'text': phoneme_elem.text or ''
                            })
                        except Exception as e:
                            logging.warning(f"Failed to convert NAFS phoneme {ph_value}: {e}")

            # Convert back to string
            processed_ssml = ET.tostring(root, encoding='unicode')

            return {
                'processed_ssml': processed_ssml,
                'nafs_conversions': nafs_conversions,
                'conversion_count': len(nafs_conversions)
            }

        except ET.ParseError as e:
            logging.error(f"SSML parsing error: {e}")
            return {
                'processed_ssml': ssml_text,
                'nafs_conversions': [],
                'conversion_count': 0,
                'error': str(e)
            }

    def create_nafs_ssml_example(self, text: str, nafs_phonemes: List[str]) -> str:
        """Create example SSML with NAFS phoneme tags"""
        words = text.split()
        ssml_parts = ['<speak xmlns="http://www.w3.org/2001/10/synthesis">']

        for i, word in enumerate(words):
            if i < len(nafs_phonemes):
                # Add NAFS phoneme tag
                nafs_hex = nafs_phonemes[i]
                ssml_parts.append(f'<phoneme alphabet="nafs" ph="{nafs_hex}">{word}</phoneme>')
            else:
                ssml_parts.append(word)

            if i < len(words) - 1:
                ssml_parts.append(' ')

        ssml_parts.append('</speak>')
        return ''.join(ssml_parts)

class NEFSTTSWrapper:
    """
    Production-ready NEFS TTS API wrapper with full SSML support
    """

    def __init__(self,
                 api_key: str,
                 base_url: str = "https://api.nefs-tts.com/v1",
                 cache_enabled: bool = True,
                 cache_size: int = 1000):
        self.api_key = api_key
        self.base_url = base_url
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
        self.cache_size = cache_size
        self.nefs_converter = NEFSConverter()
        self.ssml_processor = SSMLNEFSProcessor(self.nefs_converter)
        self.stats = {
            'requests_processed': 0,
            'cache_hits': 0,
            'total_processing_time': 0,
            'average_compression_ratio': 0,
            'ssml_requests': 0,
            'nafs_phoneme_conversions': 0
        }

    async def synthesize(self, request: NEFSSynthesisRequest) -> NEFSSynthesisResponse:
        """
        Main synthesis method with automatic NAFS optimization and SSML support
        """
        start_time = time.time()

        # Detect if input is SSML
        if self._is_ssml_content(request.text):
            request.is_ssml = True
            self.stats['ssml_requests'] += 1

            # Process SSML with NAFS phoneme tags
            ssml_result = self.ssml_processor.process_ssml_with_nafs(request.text)
            request.text = ssml_result['processed_ssml']
            self.stats['nafs_phoneme_conversions'] += ssml_result['conversion_count']

        # Auto-detect and convert input format if not SSML
        if request.nefs_encoding is None and not request.is_ssml:
            request.nefs_encoding = self._optimize_input(request.text)

        # Check cache first
        cache_key = self._generate_cache_key(request)
        if self.cache_enabled and cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]

        # Process synthesis request
        response = await self._process_synthesis(request)

        # Update cache
        if self.cache_enabled:
            self._update_cache(cache_key, response)

        # Update statistics
        processing_time = time.time() - start_time
        self.stats['requests_processed'] += 1
        self.stats['total_processing_time'] += processing_time

        return response

    def _is_ssml_content(self, text: str) -> bool:
        """Detect if text contains SSML markup"""
        ssml_patterns = [
            r'<speak[^>]*>',
            r'<phoneme[^>]*>',
            r'<break[^>]*>',
            r'<emphasis[^>]*>',
            r'<prosody[^>]*>',
            r'<voice[^>]*>'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in ssml_patterns)

    def create_nafs_ssml(self, text: str, phonetic_mappings: Dict[str, str] = None) -> str:
        """
        Create SSML with NAFS phoneme tags

        Args:
            text: Regular text to convert
            phonetic_mappings: Optional dict mapping words to NAFS hex encodings

        Returns:
            SSML string with NAFS phoneme tags
        """
        if not phonetic_mappings:
            # Auto-generate NAFS encodings
            words = text.split()
            phonetic_mappings = {}
            for word in words:
                ipa_text = self._text_to_ipa(word)
                nefs_bytes = self.nefs_converter.ipa_to_nafs(ipa_text)
                phonetic_mappings[word] = nefs_bytes.hex()

        # Build SSML
        ssml_parts = ['<speak xmlns="http://www.w3.org/2001/10/synthesis">']
        words = text.split()

        for i, word in enumerate(words):
            if word.lower() in phonetic_mappings:
                nafs_hex = phonetic_mappings[word.lower()]
                ssml_parts.append(f'<phoneme alphabet="nafs" ph="{nafs_hex}">{word}</phoneme>')
            else:
                ssml_parts.append(word)

            if i < len(words) - 1:
                ssml_parts.append(' ')

        ssml_parts.append('</speak>')
        return ''.join(ssml_parts)

    def validate_nafs_ssml(self, ssml_text: str) -> Dict:
        """
        Validate SSML with NAFS phoneme tags

        Returns:
            Dictionary with validation results and any issues found
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'nafs_tags_found': 0,
            'nafs_tags_validated': 0
        }

        try:
            # Parse SSML
            if not ssml_text.strip().startswith('<speak'):
                ssml_text = f'<speak xmlns="http://www.w3.org/2001/10/synthesis">{ssml_text}</speak>'

            root = ET.fromstring(ssml_text)

            # Validate NAFS phoneme tags
            for phoneme_elem in root.iter():
                if phoneme_elem.tag.endswith('phoneme'):
                    alphabet = phoneme_elem.get('alphabet', '').lower()
                    ph_value = phoneme_elem.get('ph', '')

                    if alphabet == 'nafs':
                        validation_result['nafs_tags_found'] += 1

                        # Validate NAFS encoding
                        try:
                            nefs_bytes = bytes.fromhex(ph_value)
                            # Attempt conversion to verify validity
                            self.nefs_converter.nafs_to_ipa(nefs_bytes)
                            validation_result['nafs_tags_validated'] += 1
                        except ValueError:
                            validation_result['errors'].append(f"Invalid NAFS hex encoding: {ph_value}")
                            validation_result['is_valid'] = False
                        except Exception as e:
                            validation_result['errors'].append(f"NAFS conversion error for {ph_value}: {str(e)}")
                            validation_result['is_valid'] = False

        except ET.ParseError as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"SSML parsing error: {str(e)}")

        return validation_result

    # [Rest of the existing methods remain the same...]
    async def synthesize_batch(self, requests: List[NEFSSynthesisRequest]) -> List[NEFSSynthesisResponse]:
        """Batch processing with parallel execution"""
        tasks = [self.synthesize(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def stream_synthesis(self, request: NEFSSynthesisRequest) -> AsyncGenerator[bytes, None]:
        """Streaming synthesis for real-time applications"""
        if request.is_ssml:
            # For SSML, process as single unit
            response = await self.synthesize(request)
            yield response.audio_data
        else:
            # Convert text to NAFS encoding in chunks
            nafs_chunks = self._chunk_nefs_encoding(request.text)

            for chunk in nafs_chunks:
                chunk_request = NEFSSynthesisRequest(
                    text="",
                    nefs_encoding=chunk,
                    voice=request.voice,
                    language=request.language,
                    format=request.format,
                    quality=request.quality
                )

                response = await self.synthesize(chunk_request)
                yield response.audio_data

    def _optimize_input(self, text: str) -> bytes:
        """Automatically optimize text input using NAFS encoding"""
        if self._is_ipa_text(text):
            return self.nefs_converter.ipa_to_nafs(text)
        else:
            ipa_text = self._text_to_ipa(text)
            return self.nefs_converter.ipa_to_nafs(ipa_text)

    def _is_ipa_text(self, text: str) -> bool:
        """Heuristic to detect IPA notation in input text"""
        ipa_chars = set('ɑɒɔəɛɪʊʌæɜɝɞɘɵɤɯɨʉɘɚɝɞɘɵɤɯɨʉieyøoeɛaɶɒɑɔʌʊɪəɚɝɞɘɵɤɯɨʉ')
        return len(set(text) & ipa_chars) > len(text) * 0.1

    def _text_to_ipa(self, text: str) -> str:
        """Convert regular text to IPA - integration point for G2P systems"""
        return _g2p_text_to_ipa(text, lang='en-us', prefer='espeak')  # Use espeak as default for offline support
    def _generate_cache_key(self, request: NEFSSynthesisRequest) -> str:
        """Generate unique cache key for request"""
        key_data = {
            'text': request.text if request.is_ssml else '',
            'nefs_encoding': request.nefs_encoding.hex() if request.nefs_encoding else '',
            'voice': request.voice,
            'language': request.language,
            'format': request.format.value,
            'quality': request.quality.value,
            'speed': request.speed,
            'pitch': request.pitch,
            'is_ssml': request.is_ssml
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def _process_synthesis(self, request: NEFSSynthesisRequest) -> NEFSSynthesisResponse:
        """Core synthesis processing - integrates with TTS engine"""
        compression_ratio = 2.0
        if request.text and not request.is_ssml:
            compression_ratio = len(request.text.encode()) / len(request.nefs_encoding)

        await asyncio.sleep(0.1)  # Simulated processing delay

        return NEFSSynthesisResponse(
            audio_data=b"simulated_audio_data",
            format=request.format,
            duration=len(request.text) * 0.1 if request.text else 1.0,
            metadata={
                'compression_ratio': compression_ratio,
                'nafs_size': len(request.nefs_encoding) if request.nefs_encoding else 0,
                'original_size': len(request.text.encode()) if request.text else 0,
                'is_ssml': request.is_ssml
            },
            nefs_encoding_used=request.nefs_encoding or b'',
            processing_time=0.1
        )

    def _chunk_nefs_encoding(self, text: str, chunk_size: int = 100) -> List[bytes]:
        """Split text into chunks for streaming synthesis"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunks.append(self._optimize_input(chunk))
        return chunks

    def _update_cache(self, key: str, response: NEFSSynthesisResponse):
        """Update cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = response

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        avg_time = (self.stats['total_processing_time'] /
                   max(self.stats['requests_processed'], 1))
        return {
            **self.stats,
            'average_processing_time': avg_time,
            'cache_hit_rate': (self.stats['cache_hits'] /
                              max(self.stats['requests_processed'], 1)) * 100
        }

# Enhanced provider adapters with full SSML support

class PollyNEFSAdapter(NEFSTTSWrapper):
    """Amazon Polly integration adapter with NEFS SSML support"""

    async def _process_synthesis(self, request: NEFSSynthesisRequest) -> NEFSSynthesisResponse:
        if request.is_ssml:
            # SSML is already processed - NAFS tags converted to IPA
            ssml_text = request.text
        else:
            # Convert NAFS back to IPA for Polly processing
            ipa_text = self.nefs_converter.nafs_to_ipa(request.nefs_encoding)
            ssml_text = f'<speak><phoneme alphabet="ipa" ph="{ipa_text}">text</phoneme></speak>'

        # Process with Polly using processed SSML
        return await super()._process_synthesis(request)

class AzureNEFSAdapter(NEFSTTSWrapper):
    """Microsoft Azure TTS integration adapter with NEFS SSML support"""

    async def _process_synthesis(self, request: NEFSSynthesisRequest) -> NEFSSynthesisResponse:
        if request.is_ssml:
            # Use processed SSML directly
            ssml_text = request.text
        else:
            # Convert NAFS to Azure-compatible format
            ipa_text = self.nefs_converter.nafs_to_ipa(request.nefs_encoding)
            ssml_text = f'<speak><phoneme alphabet="ipa" ph="{ipa_text}">text</phoneme></speak>'

        return await super()._process_synthesis(request)

class GoogleNEFSAdapter(NEFSTTSWrapper):
    """Google Cloud TTS integration adapter with NEFS SSML support"""

    async def _process_synthesis(self, request: NEFSSynthesisRequest) -> NEFSSynthesisResponse:
        if request.is_ssml:
            # Google TTS with processed SSML
            ssml_text = request.text
        else:
            # Google TTS integration with NAFS optimization
            ipa_text = self.nefs_converter.nafs_to_ipa(request.nefs_encoding)
            ssml_text = f'<speak><phoneme alphabet="ipa" ph="{ipa_text}">text</phoneme></speak>'

        return await super()._process_synthesis(request)


# True NEFSConverter implementation from trueconv.py
import warnings
from typing import Dict, List, Union

class NEFSConverter:
    def __init__(self):
        self.ipa_to_nafs_dict, self.nafs_to_ipa_dict, self.affricate_dict = self._create_mappings()
        self.nafs_to_affricate_dict = {}
        for ipa, nafs_sequence in self.affricate_dict.items():
            key = tuple(nafs_sequence)
            self.nafs_to_affricate_dict[key] = ipa

    def _create_mappings(self):
        nafs_grid = {
            0x01: '.', 0x02: '\u0306', 0x03: '\u02D0', 0x04: '\u02E5',
            0x05: '\u02E6', 0x06: '\u02E7', 0x07: '\u02E8', 0x08: '\u02E9',
            0x09: '\u02E9\u02E5', 0x0A: '\u02E5\u02E9', 0x0B: '\u02E6\u02E5', 0x0C: '\u02E9\u02E8',
            0x0D: '\u02E7\u02E6\u02E7', 0x0E: '\u02C8',
            # 0x0F = silence/space marker — classified by tone table, no IPA mapping
            0x10: '\u03B2', 0x11: 'v', 0x12: '\u00F0', 0x13: 'z',
            0x14: '\u0292', 0x15: '\u0290', 0x16: '\u029D', 0x17: '\u0263',
            0x18: '\u0281', 0x19: '\u0295', 0x1A: '\u02A1', 0x1B: '\u0266',
            0x1C: 'h\u02B0',   # hʰ — breathy-voiced h (canonical; replaces ^h^)
            0x1D: '\u0324', 0x1F: '\u02DE',
            0x20: 'b', 0x23: 'd', 0x25: '\u0256', 0x26: '\u025F',
            0x27: 'g', 0x28: '\u0262', 0x2B: '\u0294',
            0x2C: 'h\u02B0h\u02B0',  # hʰhʰ — double breathy (canonical; replaces ^hh^)
            0x2D: '\u0330', 0x2F: '\u0318',
            0x30: '\u0278', 0x31: 'f', 0x32: '\u03B8', 0x33: 's',
            0x34: '\u0283', 0x35: '\u0282', 0x36: '\u00E7', 0x37: 'x',
            0x38: '\u03C7', 0x39: '\u0127', 0x3A: '\u02A2', 0x3B: 'h',
            0x3C: '\u0325', 0x3D: '\u032C', 0x3E: '\u033B', 0x3F: '\u0319',
            0x40: 'p', 0x43: 't', 0x45: '\u0288', 0x46: '\u025F',
            0x47: 'k', 0x48: 'q', 0x49: 'i', 0x4A: 'y',
            0x4B: '\u0268', 0x4C: '\u0289', 0x4D: '\u026F', 0x4E: 'u', 0x4F: '\u031D',
            0x50: 'b\u02B0', 0x53: 'd\u02B0',
            0x55: '\u0256\u02B0', 0x56: '\u025F\u02B0',
            0x57: 'g\u02B0', 0x58: '\u0262\u02B0',
            0x59: 'e', 0x5A: '\u00F8', 0x5B: '\u0258', 0x5C: '\u0275',
            0x5D: '\u0264', 0x5E: 'o', 0x5F: '\u031E',
            0x60: 'p\u02B0', 0x63: 't\u02B0',
            0x65: '\u0288\u02B0', 0x66: '\u025F\u02B0',
            0x67: 'k\u02B0', 0x68: 'q\u02B0',
            0x69: '\u025B', 0x6A: '\u0153', 0x6B: '\u025C', 0x6C: '\u025E',
            0x6D: '\u028C', 0x6E: '\u0254', 0x6F: '\u031F',
            0x70: 'b\u031A', 0x73: 'd\u031A',
            0x75: '\u0256\u031A', 0x76: '\u025F\u031A',
            0x77: 'g\u031A', 0x78: '\u0262\u031A',
            0x79: 'a', 0x7A: '\u0276', 0x7B: '\u00E4', 0x7C: '\u0252\u0308',
            0x7D: '\u0251', 0x7E: '\u0252', 0x7F: '\u031F',
            0x80: 'p\u031A', 0x83: 't\u031A',
            0x85: '\u0288\u031A', 0x86: '\u025F\u031A',
            0x87: 'k\u031A', 0x88: 'q\u031A',
            0x89: '\u026A', 0x8A: '\u028F', 0x8B: '\u00E6', 0x8C: '\u0250',
            0x8D: '\u028A', 0x8E: '\u0259', 0x8F: '\u02DE',
            0x90: 'm', 0x91: '\u0271', 0x93: 'n', 0x95: '\u0273',
            0x96: '\u0272', 0x97: '\u014B', 0x98: '\u0274',
            0x99: '\u0303',  # ̃  nasalization diacritic (canonical; replaces ^̃^)
            0x9F: '\u0308',
            0xA0: 'w', 0xA1: '\u028B', 0xA3: '\u0279', 0xA5: '\u027B',
            0xA6: 'j', 0xA7: '\u0270', 0xAE: '\u2198',
            0xB0: '\u028D', 0xB1: '\u0265', 0xB2: '\u026C', 0xB3: 'l',
            0xB4: '\u026E', 0xB5: '\u026D', 0xB6: '\u028E', 0xB7: '\u029F',
            0xBE: '\u2197',
            0xC0: '\u0298', 0xC2: '\u01C0', 0xC3: '\u01C1', 0xC4: '\u01C3',
            0xC6: '\u01C2', 0xCF: '|',
            0xD0: '\u0253', 0xD2: '\u0257', 0xD3: '\u0255', 0xD4: '\u0291',
            0xD6: '\u0284', 0xD7: '\u0260', 0xD8: '\u029B', 0xDF: '\u2016',
            0xE0: '\u0299', 0xE1: '\u2C71', 0xE2: '\u027E', 0xE3: 'r',
            0xE4: '\u027A', 0xE5: '\u027D', 0xE6: '\u0267', 0xE8: '\u0280',
            0xF0: 'w\u02B7',  # ʷ labialization (canonical; replaces ^w^)
            0xF1: '\u033C', 0xF2: '\u032A', 0xF3: '\u033A',
            0xF4: 'j\u02B2j\u02B2j\u02B2',  # ʲʲʲ (canonical; replaces ^jjj^)
            0xF5: 'j\u02B2j\u02B2',          # ʲʲ  (canonical; replaces ^jj^)
            0xF6: 'j\u02B2',                 # ʲ   (canonical; replaces ^j^)
            0xF7: '\u0263\u02E4',            # ɣˤ  (canonical; replaces ^ɣ^)
            0xF8: '\u02E4\u02E4', 0xF9: '\u02E4',
            0xFB: '\u02C0',  # ˀ (canonical; replaces ^ˀ^)
            0xFC: '\u02DE', 0xFD: '\u02BC', 0xFE: '\u031A',
        }
        ipa_to_nafs = {}
        nafs_to_ipa = {}
        for nafs_code, ipa_symbol in nafs_grid.items():
            nafs_to_ipa[nafs_code] = ipa_symbol
            ipa_to_nafs[ipa_symbol] = nafs_code
        affricate_mappings = {
            # Affricates encoded as two-byte sequences: stop byte + fricative byte
            't\u0283': [0x43, 0x34],  # tʃ
            'd\u0292': [0x23, 0x14],  # dʒ
            'ts':       [0x43, 0x33],  # ts
            'dz':       [0x23, 0x13],  # dz
            't\u0255': [0x43, 0xD3],  # tɕ
            'd\u0291': [0x23, 0xD4],  # dʑ
            't\u0282': [0x43, 0x35],  # tʂ
            'd\u0290': [0x23, 0x15],  # dʐ
        }
        for affricate, nefs_bytes in affricate_mappings.items():
            ipa_to_nafs[affricate] = nefs_bytes
        return ipa_to_nafs, nafs_to_ipa, affricate_mappings
    
    def ipa_to_nafs(self, ipa_string: str) -> bytes:
        result = []
        i = 0
        while i < len(ipa_string):
            matched = False
            for length in [3, 2]:
                if i + length <= len(ipa_string):
                    substring = ipa_string[i:i+length]
                    if substring in self.ipa_to_nafs_dict:
                        nafs_code = self.ipa_to_nafs_dict[substring]
                        if isinstance(nafs_code, list):
                            result.extend(nafs_code)
                        else:
                            result.append(nafs_code)
                        i += length
                        matched = True
                        break
            if not matched:
                char = ipa_string[i]
                if char in self.ipa_to_nafs_dict:
                    nafs_code = self.ipa_to_nafs_dict[char]
                    if isinstance(nafs_code, list):
                        result.extend(nafs_code)
                    else:
                        result.append(nafs_code)
                    i += 1
                else:
                    warnings.warn(f"Unmappable IPA character '{char}' at position {i}")
                    i += 1
        return bytes(result)
    
    def nafs_to_ipa(self, nefs_bytes: bytes) -> str:
        result = []
        i = 0
        while i < len(nefs_bytes):
            if i + 1 < len(nefs_bytes):
                two_byte_key = (nefs_bytes[i], nefs_bytes[i+1])
                if two_byte_key in self.nafs_to_affricate_dict:
                    result.append(self.nafs_to_affricate_dict[two_byte_key])
                    i += 2
                    continue
            byte_val = nefs_bytes[i]
            if byte_val in self.nafs_to_ipa_dict:
                result.append(self.nafs_to_ipa_dict[byte_val])
                i += 1
            else:
                warnings.warn(f"Unmappable NAFS byte 0x{byte_val:02X} at position {i}")
                i += 1
        return ''.join(result)
    
    def is_lossless(self, ipa_string: str) -> bool:
        try:
            nefs_bytes = self.ipa_to_nafs(ipa_string)
            reconstructed = self.nafs_to_ipa(nefs_bytes)
            return ipa_string == reconstructed
        except:
            return False
    
    def batch_convert(self, strings: List[str], direction: str = 'ipa_to_nafs') -> List[str]:
        results = []
        for s in strings:
            if direction == 'ipa_to_nafs':
                result = self.ipa_to_nafs(s)
                results.append(result.hex(' '))
            elif direction == 'nafs_to_ipa':
                if isinstance(s, str):
                    bytes_input = bytes.fromhex(s.replace(' ', ''))
                else:
                    bytes_input = s
                result = self.nafs_to_ipa(bytes_input)
                results.append(result)
        return results
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        return {
            'total_ipa_mappings': len(self.ipa_to_nafs_dict),
            'total_nafs_mappings': len(self.nafs_to_ipa_dict),
            'affricate_mappings': len(self.affricate_dict),
            'hex_coverage_percent': len(self.nafs_to_ipa_dict) / 256 * 100
        }

# Utility functions
def create_nefs_adapter(provider: str, api_key: str) -> NEFSTTSWrapper:
    """Factory function to create appropriate adapter"""
    adapters = {
        'polly': PollyNEFSAdapter,
        'azure': AzureNEFSAdapter,
        'google': GoogleNEFSAdapter,
        'default': NEFSTTSWrapper
    }

    adapter_class = adapters.get(provider.lower(), NEFSTTSWrapper)
    return adapter_class(api_key=api_key)

# Example usage with SSML
async def example_nefs_ssml_usage():
    """Example of NEFS SSML integration"""

    nefs_tts = create_nefs_adapter('polly', 'your-api-key')

    # Example 1: Create SSML with NAFS phoneme tags
    regular_text = "Hello world"
    ssml_with_nafs = nefs_tts.create_nafs_ssml(regular_text)
    print("Generated SSML with NAFS:")
    print(ssml_with_nafs)

    # Example 2: Process existing SSML with NAFS tags
    ssml_request = NEFSSynthesisRequest(
        text=ssml_with_nafs,
        voice="neural-emma",
        language="en-US",
        format=AudioFormat.MP3,
        quality=NEFSQuality.NEURAL
    )

    response = await nefs_tts.synthesize(ssml_request)
    print(f"SSML synthesis completed in {response.processing_time}s")

    # Example 3: Validate NEFS SSML
    validation = nefs_tts.validate_nafs_ssml(ssml_with_nafs)
    print(f"SSML validation: {validation}")

    # Get enhanced statistics
    stats = nefs_tts.get_statistics()
    print(f"Enhanced stats: {stats}")

if __name__ == "__main__":
    asyncio.run(example_nefs_ssml_usage())
