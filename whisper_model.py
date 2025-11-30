import io
import numpy as np
from scipy.io import wavfile
import whisper
import os

class WhisperModel:
    def __init__(self, model_size: str = "small"):
        self.model = whisper.load_model(model_size)
        self.language = "en"

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        orig = x.dtype
        y = x.astype(np.float32)
        if orig.kind in ("i", "u"):
            scale = 32768.0 if orig.itemsize == 2 else 2147483648.0
            y /= scale
        return y

    def _to_mono(self, x: np.ndarray) -> np.ndarray:
        if getattr(x, "ndim", 1) > 1:
            return np.mean(x, axis=1)
        return x

    def _resample_16k(self, x: np.ndarray, rate: int) -> np.ndarray:
        target = 16000
        if rate == target:
            return x.astype(np.float32)
        if rate % target == 0:
            step = rate // target
            return x[::step].astype(np.float32)
        new_len = max(1, int(round(len(x) * target / float(rate))))
        x_old = np.linspace(0, len(x) - 1, num=len(x))
        x_new = np.linspace(0, len(x) - 1, num=new_len)
        return np.interp(x_new, x_old, x).astype(np.float32)

    def _prepare(self, audio: np.ndarray, rate: int) -> np.ndarray:
        mono = self._to_mono(audio)
        norm = self._normalize(mono)
        x = self._resample_16k(norm, rate)
        x = np.clip(x.astype(np.float32), -1.0, 1.0)
        return np.ascontiguousarray(x)

    def transcribe_pcm16(self, raw: bytes, sample_rate: int) -> str:
        x = np.frombuffer(raw, dtype=np.int16)
        if x.size == 0 or sample_rate <= 0:
            return ""
        x = x.astype(np.float32) / 32768.0
        if x.size < int(max(1, sample_rate * 0.5)):
            return ""
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        if rms < 2e-6:
            return ""
        x = self._resample_16k(x, sample_rate)
        if x.size == 0:
            return ""
        x = np.clip(x.astype(np.float32), -1.0, 1.0)
        x = np.ascontiguousarray(x)
        result = self.model.transcribe(x, task="translate")
        return (result.get("text") or "").strip()

    def transcribe_wav(self, raw_wav: bytes) -> str:
        rate, audio = wavfile.read(io.BytesIO(raw_wav))
        audio = self._prepare(audio, int(rate))
        result = self.model.transcribe(audio, task="translate")
        return (result.get("text") or "").strip()

    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size < int(max(1, sample_rate * 0.5)):
            return ""
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        if rms < 2e-6:
            return ""
        audio = self._prepare(audio, sample_rate)
        result = self.model.transcribe(audio, task="translate")
        return (result.get("text") or "").strip()

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="WhisperModel CLI")
    parser.add_argument("input", help="audio file path")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--type", type=str, choices=["auto", "wav", "pcm"], default="auto")
    args = parser.parse_args()
    wm = WhisperModel(args.model)
    with open(args.input, "rb") as f:
        raw = f.read()
    kind = args.type
    if kind == "auto":
        kind = "wav" if args.input.lower().endswith(".wav") else "pcm"
    if kind == "wav":
        out = wm.transcribe_wav(raw)
    else:
        out = wm.transcribe_pcm16(raw, args.sr)
    print(out)