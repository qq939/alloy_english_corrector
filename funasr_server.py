from flask import Flask, request, jsonify, render_template
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
import io
import threading
import time
import numpy as np
from scipy.io import wavfile
from funasr_model import FunASRModel
from assistant import assistant
import speech_recognition as sr
from speech_recognition import UnknownValueError
import re

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False

asr_model = FunASRModel("paraformer-zh")

buffer_text = ""
buffer_lock = threading.Lock()
speaking_active = False
speaking_last_ts = 0.0
MIN_SPEECH_RMS = 0.008
MIN_SPEECH_DURATION = 0.25

logs = []
log_once_audio = False
log_once_lock = threading.Lock()
last_submit_ts = 0.0
SUBMIT_COOLDOWN = 1.0

audio_chunks = []
audio_lock = threading.Lock()
recognizer = None

def _sanitize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    parts = re.split(r"([,\.\!])", s)
    kept = []
    for i in range(0, len(parts), 2):
        if i >= len(parts):
            break
        body = parts[i].strip()
        if not body:
            continue
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if re.search(r"\bthank\s+you\b", body, flags=re.I):
            continue
        kept.append(body + (punct or ""))
    s = " ".join(kept).strip()
    s = re.sub(r"(\b\w+\b)(?:\s+\1\b){1,}", r"\1", s, flags=re.I)
    s = re.sub(r"(\b\w+\b[\.\!\?])(?:\s+\1){1,}", r"\1", s, flags=re.I)
    return s

def add_log(s):
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="ignore")
        except Exception:
            s = s.decode("latin-1", errors="ignore")
    else:
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n")
    ts = time.strftime("%H:%M:%S")
    logs.append(f"[{ts}] {s}")
    if len(logs) > 200:
        del logs[: len(logs) - 200]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/logs")
def get_logs():
    return jsonify({"logs": logs[-100:]})

@app.route("/api/audio", methods=["POST"]) 
def upload_audio():
    global buffer_text, speaking_active, speaking_last_ts, last_submit_ts, recognizer
    file_raw = request.files.get("audio_raw")
    file_wav = request.files.get("audio")
    if not file_raw and not file_wav:
        return jsonify({"ok": False, "error": "no audio"}), 400
    raw = (file_raw or file_wav).read()
    with log_once_lock:
        global log_once_audio
        if not log_once_audio:
            add_log("音频片段已接收")
            log_once_audio = True
    now_ts = time.time()
    sr_rate = 16000
    arr = None
    try:
        if file_raw:
            try:
                sr_rate = int(request.form.get("sr", "16000"))
            except Exception:
                sr_rate = 16000
            if sr_rate <= 0 or sr_rate < 8000 or sr_rate > 96000:
                return jsonify({"ok": True, "text": ""})
            if not raw:
                return jsonify({"ok": True, "text": ""})
            nsamples = len(raw) // 2
            min_samples = int(max(1600, sr_rate * 0.3))
            if nsamples < min_samples:
                return jsonify({"ok": True, "text": ""})
            if len(raw) % 2 != 0:
                raw = raw[: len(raw) - 1]
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            arr = x
            pcm_bytes = raw
        else:
            rate, audio_np = wavfile.read(io.BytesIO(raw))
            try:
                sr_rate = int(rate)
            except Exception:
                sr_rate = 16000
            if sr_rate <= 0 or sr_rate < 8000 or sr_rate > 96000:
                return jsonify({"ok": True, "text": ""})
            if getattr(audio_np, "ndim", 1) > 1:
                audio_np = np.mean(audio_np, axis=1)
            if audio_np.dtype.kind in ("i", "u"):
                x = audio_np.astype(np.float32) / 32768.0
            else:
                x = audio_np.astype(np.float32)
            arr = x
            y = np.clip(x, -1.0, 1.0)
            pcm_bytes = (y * 32767.0).astype(np.int16).tobytes()
        dur = len(arr) / float(sr_rate) if arr is not None else 0.0
        rms = float(np.sqrt(np.mean(np.square(arr)))) if arr is not None and arr.size else 0.0
        if (not speaking_active) and (dur < MIN_SPEECH_DURATION and rms < MIN_SPEECH_RMS):
            return jsonify({"ok": True, "text": ""})
        if not getattr(asr_model, "_streaming", False):
            asr_model.start_stream(sr_rate)
        asr_model.push_array(arr, sr_rate)
        with audio_lock:
            audio_chunks.append(arr)
        speaking_active = True
        speaking_last_ts = now_ts
        if recognizer is None:
            recognizer = sr.Recognizer()
        try:
            ad = sr.AudioData(pcm_bytes, sr_rate, 2)
            phrase = recognizer.recognize_whisper(ad, model="base", language="english")
        except UnknownValueError:
            phrase = ""
        b = (phrase or "").strip().lower()
        ends_ok = re.search(r"(?:^|\s)(ok|okay)[\.!?\"]?$", b)
        if ((b in {"ok", "okay"}) or ends_ok) and (now_ts - last_submit_ts) > SUBMIT_COOLDOWN:
            with audio_lock:
                if audio_chunks:
                    full = np.concatenate(audio_chunks).astype(np.float32)
                    audio_chunks.clear()
                else:
                    full = arr.astype(np.float32)
            full = np.clip(full, -1.0, 1.0)
            full = np.ascontiguousarray(full, dtype=np.float32)
            _t0 = time.time()
            text = asr_model.finish_stream()
            dt = time.time() - _t0
            if text:
                text = _sanitize_text(text)
                add_log(f"识别完成（用时{dt:.2f}秒）")
                final = re.sub(r"\s*(ok|okay)[\.!?\"]?$", "", text.lower(), flags=re.IGNORECASE).strip()
                resp = assistant.answer(final, None)
                add_log("提交已触发")
                if resp:
                    added = False
                    if isinstance(resp, bytes):
                        try:
                            text_resp = resp.decode("utf-8", errors="ignore")
                        except Exception:
                            text_resp = resp.decode("latin-1", errors="ignore")
                    else:
                        text_resp = str(resp)
                    text_resp = text_resp.replace("\r\n", "\n").replace("\r", "\n")
                    text_resp = text_resp.replace("\\n", "\n")
                    for ln in text_resp.splitlines():
                        ln = ln.strip()
                        if ln:
                            add_log(ln)
                            added = True
                    if not added:
                        add_log(text_resp)
                with buffer_lock:
                    buffer_text = ""
                last_submit_ts = now_ts
            return jsonify({"ok": True, "text": text or ""})
    except Exception as e:
        add_log(f"ASR识别异常: {e}")
        return jsonify({"ok": True, "text": ""})
    return jsonify({"ok": True, "text": ""})

@app.route("/api/recognize", methods=["POST"]) 
def recognize_now():
    global last_submit_ts
    now_ts = time.time()
    with audio_lock:
        if not audio_chunks:
            return jsonify({"ok": True, "text": ""})
        full = np.concatenate(audio_chunks).astype(np.float32)
        audio_chunks.clear()
    full = np.clip(full, -1.0, 1.0)
    full = np.ascontiguousarray(full, dtype=np.float32)
    try:
        _t0 = time.time()
        text = asr_model.finish_stream()
        dt = time.time() - _t0
    except Exception as e:
        add_log(f"ASR识别异常: {e}")
        return jsonify({"ok": True, "text": ""})
    if not text:
        return jsonify({"ok": True, "text": ""})
    text = _sanitize_text(text)
    add_log(f"识别完成（用时{dt:.2f}秒）")
    final = text.lower()
    resp = assistant.answer(final, None)
    add_log("提交已触发")
    if resp:
        added = False
        if isinstance(resp, bytes):
            try:
                text_resp = resp.decode("utf-8", errors="ignore")
            except Exception:
                text_resp = resp.decode("latin-1", errors="ignore")
        else:
            text_resp = str(resp)
        text_resp = text_resp.replace("\r\n", "\n").replace("\r", "\n")
        text_resp = text_resp.replace("\\n", "\n")
        for ln in text_resp.splitlines():
            ln = ln.strip()
            if ln:
                add_log(ln)
                added = True
        if not added:
            add_log(text_resp)
    with buffer_lock:
        buffer_text = ""
    last_submit_ts = now_ts
    return jsonify({"ok": True, "text": text})

def run():
    port = int(os.getenv("PORT", "5010"))
    try:
        _warm = np.zeros(16000, dtype=np.float32)
        try:
            asr_model.transcribe_array(_warm, 16000)
        except Exception:
            pass
    except Exception:
        pass
    try:
        app.run(host="0.0.0.0", port=port)
    except OSError:
        alt = 5001 if port == 5000 else 8000
        app.run(host="0.0.0.0", port=alt)

if __name__ == "__main__":
    run()
