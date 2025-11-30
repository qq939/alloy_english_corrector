from flask import Flask, request, jsonify, render_template, send_file
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
import speech_recognition as sr
from assistant import assistant

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False

recognizer = sr.Recognizer()

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
SUBMIT_COOLDOWN = 1.0

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
    text = ""
    try:
        if file_raw:
            try:
                sr_rate = int(request.form.get("sr", "16000"))
            except Exception:
                sr_rate = 16000
            audio_data = sr.AudioData(raw, sr_rate, 2)
        else:
            with sr.AudioFile(io.BytesIO(raw)) as src:
                audio_data = recognizer.record(src)
        try:
            text = recognizer.recognize_whisper(audio_data, model="base", language="english")
        except (sr.UnknownValueError, sr.RequestError, Exception) as e:
            add_log(f"识别异常: {e}")
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
            except Exception as e2:
                add_log(f"识别回退失败: {e2}")
                text = ""
    except Exception as e:
        add_log(f"音频解析失败: {e}")
        text = ""
    if text:
        text = _sanitize_text(text)
    if not text:
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
        silent = sr.AudioData(b"\x00" * (16000 * 2 // 2), 16000, 2)
        try:
            recognizer.recognize_whisper(silent, model="base", language="english")
        except Exception:
            pass
    except Exception:
        pass
    try:
        app.run(host="0.0.0.0", port=port, ssl_context=ssl_context)
    except OSError:
        alt = 5001 if port == 5000 else 8000
        app.run(host="0.0.0.0", port=alt, ssl_context=ssl_context)

if __name__ == "__main__":
    run()