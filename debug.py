import subprocess
import sys

ESPEAK_BIN = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

# Test 1: bare version check
print("=== Version ===")
r = subprocess.run([ESPEAK_BIN, "--version"], capture_output=True)
print("stdout:", r.stdout[:200])
print("stderr:", r.stderr[:200])
print("returncode:", r.returncode)

# Test 2: synthesize to stdout
print("\n=== Synth to stdout ===")
r = subprocess.run(
    [ESPEAK_BIN, "--stdout", "hello"],
    capture_output=True, timeout=5
)
print("stdout length:", len(r.stdout))
print("stdout[:20]:", r.stdout[:20])
print("stderr:", r.stderr[:200])
print("returncode:", r.returncode)

# Test 3: synthesize IPA to stdout
print("\n=== IPA to stdout ===")
r = subprocess.run(
    [ESPEAK_BIN, "--stdout", "--ipa=0", "[[hɛloʊ]]"],
    capture_output=True, timeout=5
)
print("stdout length:", len(r.stdout))
print("stdout[:20]:", r.stdout[:20])
print("stderr:", r.stderr[:200])
print("returncode:", r.returncode)