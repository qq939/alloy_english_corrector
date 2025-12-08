
import sys
import os
import logging
import threading
import time
import re
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from RealtimeSTT import AudioToTextRecorder
from assistant import assistant
import ssl

# --- Configuration ---
# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.config["JSON_AS_ASCII"] = False

# Logging Setup
# We want to maintain a list of logs for the frontend, similar to whisper_server.py
server_logs = []
log_lock = threading.Lock()

def add_server_log(s):
    global server_logs
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="ignore")
        except Exception:
            s = s.decode("latin-1", errors="ignore")
    else:
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    ts = time.strftime("%H:%M:%S")
    with log_lock:
        server_logs.append(f"[{ts}] {s}")
        if len(server_logs) > 200:
            del server_logs[: len(server_logs) - 200]

# Configure standard logging to also output to our internal log list
class ListHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        # Avoid duplicate timestamps in our internal log if possible, 
        # but standard logging format might include them.
        # Let's just use the message part if we can, or just log it.
        # Simple approach: Log the message.
        # We strip the timestamp from the formatter if we want consistent internal logs,
        # but here we just append what we get.
        pass # We'll use add_server_log explicitly for application events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realtime_server")
# Remove existing handlers
for h in logger.handlers[:]:
    logger.removeHandler(h)

file_handler = logging.FileHandler("realtime_server.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

# --- Global State ---
recorder = None
recorder_ready = threading.Event()
full_transcript = []  # List of committed sentences
current_partial = ""  # Current streaming text
buffer_lock = threading.Lock()
submissions = [] # For download
last_submit_lines = [] # For logs endpoint (assistant response history)

# --- Callbacks ---

def on_realtime_update(text):
    """Called when the partial transcription updates."""
    global current_partial
    with buffer_lock:
        current_partial = text

def on_vad_start():
    logger.info("VAD detected speech start.")

def on_vad_stop():
    logger.info("VAD detected speech stop.")

# --- Background Worker ---

def transcription_loop():
    """Continuously retrieves committed text from the recorder."""
    global recorder, full_transcript, current_partial, submissions, last_submit_lines
    logger.info("Transcription loop started.")
    add_server_log("Transcription loop started")
    
    while True:
        try:
            # text() blocks until a sentence is completed (silence detected)
            text = recorder.text()
            if text.strip():
                logger.info(f"Committed: {text}")
                add_server_log(f"Recognized: {text}")
                
                should_submit = False
                text_to_submit = ""
                
                with buffer_lock:
                    full_transcript.append(text.strip())
                    current_partial = ""
                    
                    # Check for "ok" trigger
                    all_text = " ".join(full_transcript)
                    if _check_is_ok(all_text):
                        should_submit = True
                        # Consume buffer
                        full_transcript = []
                        text_to_submit = _strip_ok_suffix(all_text)
                
                if should_submit and text_to_submit:
                    # Trigger AI Assistant
                    threading.Thread(target=process_ai_response, args=(text_to_submit,)).start()
                
        except Exception as e:
            logger.error(f"Error in transcription loop: {e}")
            time.sleep(1)

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

def _check_is_ok(text):
    if not text:
        return False
    text = text.lower()
    ends_ok = re.search(r"(?:^|\s)(ok|okay)[\.!?\"]*$", text)
    return (" ok" in text) or (" okay" in text) or ends_ok

def _strip_ok_suffix(text):
    return re.sub(r"\s*(ok|okay)[\.!?\"]?$", "", text, flags=re.IGNORECASE).strip()

def process_ai_response(text):
    global submissions, last_submit_lines
    try:
        add_server_log("Submitting to AI...")
        
        # Sanitize text
        text = _sanitize_text(text)
        final = text.lower()
        
        # Call assistant
        resp = assistant.answer(final, None)
        add_server_log("提交已触发")
        
        timestamp = time.time()
        
        lines_for_display = []
        assistant_text = ""
        
        if resp:
            if isinstance(resp, bytes):
                try:
                    text_resp = resp.decode("utf-8", errors="ignore")
                except Exception:
                    text_resp = resp.decode("latin-1", errors="ignore")
            else:
                text_resp = str(resp)
            
            text_resp = text_resp.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
            assistant_text = text_resp.strip()
            
            added = False
            for ln in text_resp.splitlines():
                ln = ln.strip()
                if ln:
                    add_server_log(ln)
                    lines_for_display.append(ln)
                    added = True
            
            if not added:
                add_server_log(text_resp)
                lines_for_display = [text_resp]
        else:
            lines_for_display = [text]

        with buffer_lock:
            submissions.append({
                "ts": timestamp,
                "recognized": text,
                "assistant": assistant_text
            })
            last_submit_lines = lines_for_display
            
    except Exception as e:
        logger.error(f"AI Assistant error: {e}")
        add_server_log(f"AI Error: {e}")

# --- Initialization ---

def init_recorder():
    global recorder
    if recorder is not None:
        return

    logger.info("Initializing RealtimeSTT...")
    add_server_log("Initializing RealtimeSTT Model...")
    
    # Parameters tuned for better VAD and streaming response
    recorder = AudioToTextRecorder(
        model="small", 
        language="en",
        use_microphone=False, # We feed audio manually
        enable_realtime_transcription=True,
        on_realtime_transcription_update=on_realtime_update,
        on_vad_start=on_vad_start,
        on_vad_stop=on_vad_stop,
        
        # VAD & Timing Parameters
        silero_sensitivity=0.4,
        post_speech_silence_duration=0.7, 
        min_length_of_recording=0.5,
        min_gap_between_recordings=0,
        realtime_processing_pause=0.05, 
        realtime_model_type="tiny",
    )
    
    logger.info("RealtimeSTT Initialized.")
    add_server_log("RealtimeSTT Ready.")
    recorder_ready.set()
    
    # Start the loop thread
    t = threading.Thread(target=transcription_loop, daemon=True)
    t.start()

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/audio", methods=["POST"])
def api_audio():
    if not recorder_ready.is_set():
        return jsonify({"ok": False, "error": "Server starting up..."}), 503

    try:
        file_raw = request.files.get("audio_raw")
        sr_str = request.form.get("sr", "16000")
        try:
            sr = int(sr_str)
        except:
            sr = 16000

        if file_raw:
            raw_data = file_raw.read()
            # Convert to numpy int16 array for feed_audio to handle resampling if needed
            audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
            recorder.feed_audio(audio_int16, original_sample_rate=sr)
        
        return jsonify({"ok": True})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/logs")
def api_logs():
    with buffer_lock:
        live = " ".join(full_transcript)
        if current_partial:
            live += " " + current_partial
        live = live.strip()
        # In whisper_server, 'logs' in json was 'last_submit_lines' (the AI response).
        # But there was also a 'server logs' concept. 
        # Looking at app.js: 
        #   liveBox.textContent = d.live
        #   logsBox.innerHTML = (d.logs || []).map...
        # In whisper_server.py: 
        #   return jsonify({"logs": last_submit_lines, "live": live})
        # So 'logs' here is actually the AI response display, NOT the debug logs.
        # Wait, whisper_server.py line 346: return jsonify({"logs": last_submit_lines, "live": live})
        # And last_submit_lines is set in submit_text.
        # However, there is also 'add_log' in whisper_server.py which appends to global 'logs' list.
        # Does app.js fetch that?
        # app.js fetchLogs() calls /api/logs.
        # So d.logs corresponds to last_submit_lines.
        # BUT wait, app.js line 234: logsBox.innerHTML = (d.logs || []).map...
        # In whisper_server.py line 329: last_submit_lines = lines_for_display.
        # And lines_for_display includes the user text and AI response.
        
        # So we should return the formatted AI conversation in "logs".
        logs_response = last_submit_lines
        
    return jsonify({
        "live": live,
        "logs": logs_response
    })

@app.route("/api/recognize", methods=["POST"])
def recognize_now():
    """
    Manually trigger recognition/submission.
    Consumes all accumulated text (full_transcript + current_partial) and submits to AI.
    """
    global current_partial, full_transcript
    
    text_to_submit = ""
    with buffer_lock:
        parts = list(full_transcript)
        if current_partial:
            parts.append(current_partial)
        
        text_to_submit = " ".join(parts).strip()
        
        # Clear buffers
        full_transcript = []
        current_partial = ""
    
    if text_to_submit:
        add_server_log(f"Manual Recognize: {text_to_submit}")
        # Run AI response in background
        threading.Thread(target=process_ai_response, args=(text_to_submit,)).start()
        return jsonify({"ok": True, "text": text_to_submit})
    
    return jsonify({"ok": True, "text": ""})

@app.route("/reset", methods=["POST"])
def reset():
    global full_transcript, current_partial, last_submit_lines
    with buffer_lock:
        full_transcript = []
        current_partial = ""
        last_submit_lines = []
        recorder.stop()
        recorder.start()
    
    # We also probably want to clear RealtimeSTT's buffer if possible, 
    # but the API doesn't expose it easily. 
    # recorder.stop() then recorder.start() might work but is heavy.
    # For now, just clearing visual buffers is usually what's requested.
    
    add_server_log("Buffer reset requested.")
    return jsonify({"status": "ok"})

@app.route("/api/download_submissions")
def download_submissions():
    try:
        parts = []
        for item in submissions:
            ts = item.get("ts")
            rec = str(item.get("recognized", ""))
            asst = str(item.get("assistant", ""))
            header = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}]" if ts else ""
            block = [header]
            if rec:
                block.append(f"识别: {rec}")
            if asst:
                block.append(f"提交: {asst}")
            block.append("-" * 40)
            parts.append("\n".join([x for x in block if x]))
        content = "\n\n".join(parts)
    except Exception:
        content = ""
    
    resp = Response(content, mimetype="text/plain; charset=utf-8")
    resp.headers["Content-Disposition"] = "attachment; filename=submissions.txt"
    return resp

if __name__ == "__main__":
    init_recorder()
    
    port = int(os.getenv("PORT", "5010"))
    
    # SSL Support matching whisper_server.py
    cert = os.getenv("TLS_CERT", "server.crt")
    key = os.getenv("TLS_KEY", "server.key")
    
    ssl_context = None
    if os.path.exists(cert) and os.path.exists(key):
        try:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(cert, key)
            logger.info(f"SSL Context loaded from {cert} and {key}")
        except Exception as e:
            logger.error(f"Failed to load SSL: {e}")

    # Disable reloader to prevent double initialization of heavy models
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False, ssl_context=ssl_context)