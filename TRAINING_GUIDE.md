# NEFS TTS Training Guide

This guide will help you train the NEFS-based Text-to-Speech system with HiFi-GAN.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- **RAM**: At least 16GB system RAM
- **Storage**: 50GB+ free space for dataset and checkpoints

### Software Requirements
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install torchaudio
pip install numpy
pip install librosa
pip install matplotlib
pip install tensorboard  # for monitoring training
```

## Step 1: Prepare Your Dataset

You need a speech dataset with aligned text. Popular options:

### Option A: LJSpeech (Recommended for beginners)
```bash
# Download LJSpeech dataset
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvjf LJSpeech-1.1.tar.bz2
```

### Option B: Your own dataset
Organize your data in this structure:
```
data/
├── wavs/
│   ├── audio001.wav
│   ├── audio002.wav
│   └── ...
└── metadata.csv  # Format: filename|text
```

## Step 2: Modify the Training Script

Edit `train_nefs_tts.py` to point to your dataset:

```python
# Line ~280 in train_nefs_tts.py
data_dir = "path/to/LJSpeech-1.1"  # Change this to your dataset path
```

## Step 3: Start Training

### Basic Training Command
```bash
python train_nefs_tts.py
```

### Training with Custom Parameters
```bash
python train_nefs_tts.py \
  --data_dir /path/to/dataset \
  --checkpoint_dir ./checkpoints \
  --batch_size 16 \
  --learning_rate 0.0002 \
  --num_epochs 1000
```

### Monitor Training with TensorBoard
```bash
# In a separate terminal
tensorboard --logdir=./runs
```
Then open http://localhost:6006 in your browser

## Step 4: Training Process

The training will:
1. **Load and preprocess audio** → Convert to NEFS phonemes
2. **Train generator** → Learn to synthesize speech from NEFS embeddings
3. **Train discriminators** → Ensure realistic audio quality
4. **Save checkpoints** → Every 1000 steps in `./checkpoints/`

### Expected Training Time
- **Small dataset (LJSpeech ~24 hours)**: 2-3 days on RTX 3080
- **Large dataset**: 1-2 weeks

### Key Metrics to Watch
- **Generator Loss**: Should decrease and stabilize
- **Discriminator Loss**: Should oscillate around 0.3-0.7
- **Audio Quality**: Check sample outputs periodically

## Step 5: Resume Training (if interrupted)

```python
# In train_nefs_tts.py, uncomment and modify:
checkpoint = torch.load('checkpoints/checkpoint_step_10000.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_step = checkpoint['step']
```

## Step 6: Test Your Model

Create a test script `test_tts.py`:

```python
import torch
from nefs_tts_hifigan import NEFSTTSSynthesizer
import soundfile as sf

# Load trained model
model = NEFSTTSSynthesizer()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate speech
text = "Hello world, this is NEFS text to speech."
audio = model.synthesize(text)

# Save audio
sf.write('output.wav', audio, 22050)
```

Run:
```bash
python test_tts.py
```

## Troubleshooting

### Out of Memory Errors
- **Reduce batch size**: Edit `batch_size` in training script (try 8 or 4)
- **Reduce audio length**: Filter out long audio files
- **Use gradient accumulation**: Accumulate gradients over multiple batches

### Poor Audio Quality
- **Train longer**: Need at least 50k-100k steps for decent quality
- **Check dataset**: Ensure clean, high-quality audio
- **Adjust discriminator**: Balance generator/discriminator training

### CUDA Errors
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version
```

### NaN Losses
- **Lower learning rate**: Try 0.0001 instead of 0.0002
- **Gradient clipping**: Add `torch.nn.utils.clip_grad_norm_(params, 1.0)`
- **Check data normalization**: Ensure audio is properly normalized

## Advanced: Real-Time Synthesis

Once trained, you can use the model for real-time synthesis:

```python
import torch
from nefs_tts_hifigan import NEFSTTSSynthesizer

class RealtimeTTS:
    def __init__(self, checkpoint_path):
        self.model = NEFSTTSSynthesizer().cuda()
        self.model.load_state_dict(torch.load(checkpoint_path)['model'])
        self.model.eval()
    
    def speak(self, text):
        with torch.no_grad():
            audio = self.model.synthesize(text)
        return audio

# Usage
tts = RealtimeTTS('checkpoints/best_model.pt')
audio = tts.speak("Your text here")
```

## Next Steps

1. **Fine-tune on your voice**: Record 1-2 hours of your own speech
2. **Multi-speaker**: Modify model to support multiple voices
3. **Prosody control**: Add emotion/style controls
4. **Web interface**: Integrate with Flask/FastAPI for web demo
5. **Optimize inference**: Use ONNX or TorchScript for faster synthesis

## Estimated Resource Usage

| Component | Training | Inference |
|-----------|----------|----------|
| GPU Memory | 6-8 GB | 2-4 GB |
| RAM | 16 GB | 8 GB |
| Disk I/O | High | Low |
| Time/epoch | 30-60 min | <1 sec/sentence |

## Questions?

If you encounter issues:
1. Check the error logs carefully
2. Verify dataset format matches expectations
3. Ensure all dependencies are installed
4. Try with a smaller batch size first

Good luck with your training!
