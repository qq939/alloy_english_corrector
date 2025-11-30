from flask import Flask, request, jsonify, render_template, send_file
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass
import base64
import io
import threading
import time
import numpy as np
from scipy.io import wavfile
import whisper
from assistant import assistant

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False

whisper_model = whisper.load_model("small")

buffer_text = ""
buffer_lock = threading.Lock()
speaking_active = False
speaking_last_ts = 0.0
MIN_SPEECH_RMS = 0.008
MIN_SPEECH_DURATION = 0.25

logs = []
log_once_audio = False
log_once_recog = False
log_once_lock = threading.Lock()
last_submit_ts = 0.0
SUBMIT_COOLDOWN = 2.0

import re

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
    global buffer_text, speaking_active, speaking_last_ts, last_submit_ts
    file = request.files.get("audio")
    if not file:
        return jsonify({"ok": False, "error": "no audio"}), 400
    raw = file.read()
    with log_once_lock:
        global log_once_audio
        if not log_once_audio:
            add_log("音频片段已接收")
            log_once_audio = True
    rate, audio_np = wavfile.read(io.BytesIO(raw))
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    audio_np = audio_np.astype(np.float32) / 32768.0
    duration = len(audio_np) / float(rate)
    rms = float(np.sqrt(np.mean(np.square(audio_np))))
    now_ts = time.time()
    if (not speaking_active) and (duration < MIN_SPEECH_DURATION and rms < MIN_SPEECH_RMS):
        return jsonify({"ok": True, "text": ""})
    result = whisper_model.transcribe(
        audio_np,
        no_speech_threshold=0.5,
        logprob_threshold=-1.1,
        compression_ratio_threshold=2.0,
        temperature=0.0,
        condition_on_previous_text=False,
        prompt='Translate mixed Chinese and English speech to English. Allow normal pauses. Return empty when no speech. Do not add irrelevant text like ["Thanks for watching!", "謝謝收看"].',
        language=None,
        fp16=False,
        task="translate",
        verbose=None,
    )
    text = (result.get("text") or "").strip()
    if text:
        text = _sanitize_text(text)
    if not text:
        if rms >= MIN_SPEECH_RMS:
            speaking_active = True
            speaking_last_ts = now_ts
        elif speaking_active and (now_ts - speaking_last_ts) > 2.0:
            speaking_active = False
        return jsonify({"ok": True, "text": ""})
    with buffer_lock:
        buffer_text = (buffer_text + " " + text).strip()
        b = buffer_text.lower()
    with log_once_lock:
        global log_once_recog
        if not log_once_recog:
            add_log("识别完成")
            log_once_recog = True
    speaking_active = True
    speaking_last_ts = now_ts
    import re
    ends_ok = re.search(r"(?:^|\s)(ok|okay)[\.!?\"]?$", b)
    if ((text.strip().lower() in {"ok", "okay"}) or ends_ok) and (now_ts - last_submit_ts) > SUBMIT_COOLDOWN:
        final = re.sub(r"\s*(ok|okay)[\.!?\"]?$", "", b, flags=re.IGNORECASE).strip()
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
    ssl_mode = os.getenv("SSL", "")
    ssl_cert = os.getenv("SSL_CERT")
    ssl_key = os.getenv("SSL_KEY")
    ssl_context = None
    if ssl_mode.lower() == "adhoc":
        ssl_context = "adhoc"
    elif ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
    try:
        app.run(host="0.0.0.0", port=port, ssl_context=ssl_context)
    except OSError:
        alt = 5001 if port == 5000 else 8000
        app.run(host="0.0.0.0", port=alt, ssl_context=ssl_context)

if __name__ == "__main__":
    run()