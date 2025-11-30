import io
import os
import tempfile
import numpy as np
from scipy.io import wavfile

class WhisperModel:    # 实际上是FunASR，为了和Whisper模型保持一致，所以叫WhisperModel
    def __init__(self, model_name: str = "paraformer-zh", vad_model: str | None = None, punc_model: str | None = None, device: str | None = None):
        try:
            if model_name == "small":
                model_name = "paraformer-zh"
            from funasr import AutoModel
            self._auto = AutoModel(model=model_name, vad_model=vad_model, punc_model=punc_model, device=device)
            self._init_error = None
        except Exception as e:
            self._auto = None
            self._init_error = e

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

    def _ensure_ready(self):
        if self._auto is None:
            raise RuntimeError(f"funASR未初始化: {self._init_error}")

    def _extract_text(self, res):
        if isinstance(res, dict):
            t = res.get("text") or res.get("sentence") or ""
            return (t or "").strip()
        if isinstance(res, (list, tuple)):
            item = res[0] if res else {}
            t = getattr(item, "get", lambda k, d=None: None)("text") or getattr(item, "get", lambda k, d=None: None)("sentence") or ""
            return (t or "").strip()
        return (str(res) if res else "").strip()

    def _gen_from_path(self, path: str) -> str:
        self._ensure_ready()
        out = self._auto.generate(input=path)
        return self._extract_text(out)

    def transcribe_pcm16(self, raw: bytes, sample_rate: int) -> str:
        if not raw or sample_rate <= 0:
            return ""
        x = np.frombuffer(raw, dtype=np.int16)
        if x.size < int(max(1, sample_rate * 0.5)):
            return ""
        rms = float(np.sqrt(np.mean(np.square(x.astype(np.float32))))) if x.size else 0.0
        if rms < 2e-6:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, sample_rate, x)
            path = tmp.name
        try:
            return self._gen_from_path(path)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def transcribe_wav(self, raw_wav: bytes) -> str:
        rate, audio = wavfile.read(io.BytesIO(raw_wav))
        audio = np.asarray(audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, int(rate), audio.astype(np.int16) if audio.dtype != np.int16 else audio)
            path = tmp.name
        try:
            return self._gen_from_path(path)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> str:
        if getattr(audio, "size", 0) < int(max(1, sample_rate * 0.5)):
            return ""
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32))))) if audio.size else 0.0
        if rms < 2e-6:
            return ""
        x = self._prepare(audio, sample_rate)
        y = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, 16000, y)
            path = tmp.name
        try:
            return self._gen_from_path(path)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass