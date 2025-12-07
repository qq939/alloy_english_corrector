from flask import Flask, request, jsonify, render_template
import os
import multiprocessing as mp
import io
import threading
import time
import numpy as np
from scipy.io import wavfile
import re
import ssl
import logging
from whisper_streaming import WhisperStreamingModel
from assistant import assistant

# Configuration
os.environ.setdefault("TQDM_DISABLE", "1")
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False

# Logging Setup
handler = logging.FileHandler("whisper_server.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(message)s %(asctime)s"))
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)
app.logger.debug("Whisper Server Starting...")

# Global State
asr_model = WhisperStreamingModel("small")
buffer_text = ""
buffer_lock = threading.Lock()

last_chunk_audio = None
last_chunk_sr = 16000
last_partial_text = ""
last_partial_ts = 0.0  # For throttling
PARTIAL_INTERVAL = 1 # Throttling interval (seconds)

speaking_active = False
speaking_last_ts = 0.0
MIN_SPEECH_RMS = 0.08
MIN_SPEECH_DURATION = 0.25

logs = []
log_once_audio = False
log_once_lock = threading.Lock()

last_submit_ts = 0.0
SUBMIT_COOLDOWN = 2.0
submissions = []
last_submit_lines = []

# Helper Functions
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

def _append_chunk(base: str, prev_text: str, cur_text: str) -> str:
    """
    基于滑动窗口的文本拼接逻辑。
    核心目标：找到 prev_text 的后缀与 cur_text 的前缀之间的最大重叠，
    然后将 cur_text 中不重叠的后缀部分追加到 base。
    """
    cur = (cur_text or "").strip()
    prev = (prev_text or "").strip()
    
    if not cur:
        return base or ""
    
    if not prev:
        # 如果没有前一段文本，直接追加当前文本
        return (base + (" " if base else "") + cur).strip()

    # 1. 简单的包含关系检查
    # 如果当前文本完全包含在前一段中（例如说话停顿，窗口只移动了一点点，识别结果可能变短或不变）
    # 或者前一段完全包含在当前文本中（例如窗口刚开始扩大）
    # 但对于滑动窗口，通常是部分重叠。
    
    # 为了比较鲁棒性，去除标点和大小写进行重叠计算
    def normalize(s):
        return re.sub(r"[^\w\s]", "", s).lower().replace("\s+", "")

    # 我们从可能的最大重叠长度开始尝试，直到0
    # 重叠长度最多是 len(prev) 和 len(cur) 的较小值
    n = min(len(prev), len(cur))
    best_overlap_len = 0
    
    # 暴力搜索最大后缀-前缀重叠 (Longest Suffix-Prefix Overlap)
    # 优化：只检查最后N个字符，避免全文搜索
    # 实际上对于几秒的文本，直接遍历开销很小
    
    # 这里的逻辑是：prev 的后 k 个字符 == cur 的前 k 个字符
    for k in range(n, 0, -1):
        # 这是一个严格匹配，为了更好的效果，可能需要 fuzzy match，
        # 但严格匹配对于避免重复最安全。
        # 我们比较 prev 的后缀和 cur 的前缀
        if prev.endswith(cur[:k]):
            best_overlap_len = k
            break
            
    # 如果严格匹配失败，尝试用标准化后的文本再试一次（处理标点抖动问题）
    if best_overlap_len == 0 and n > 3:
        # 定义简单的标准化：小写 + 去除标点和空格，只保留字母数字
        def clean(s): return re.sub(r"[^\w]", "", s).lower()
        c_prev = clean(prev)
        c_cur = clean(cur)
        
        nn = min(len(c_prev), len(c_cur))
        norm_overlap_len = 0
        
        # 寻找归一化后的最大重叠
        for k in range(nn, 0, -1):
            if c_prev.endswith(c_cur[:k]):
                norm_overlap_len = k
                break
        
        # 如果找到了归一化重叠，需要映射回原始字符串 cur 的索引
        if norm_overlap_len > 0:
            current_norm_len = 0
            real_idx = 0
            for char in cur:
                # 这里的逻辑必须与 clean 函数一致：如果是有效字符则计数
                if re.match(r"[\w]", char): 
                    current_norm_len += 1
                real_idx += 1 # 原始索引前移
                
                # 当有效字符计数达到重叠长度时，当前的 real_idx 就是切分点
                if current_norm_len == norm_overlap_len:
                    best_overlap_len = real_idx
                    break

    # 计算增量
    if best_overlap_len > 0:
        delta = cur[best_overlap_len:].strip()
    else:
        # 如果完全没找到重叠（可能是因为Whisper识别结果跳变太大，或者真的没重叠）
        # 这种情况下直接拼接会有风险，但如果不拼就会丢字。
        # 对于滑动窗口，如果没有重叠，通常意味着 cur 是全新的（窗口跳跃？）或者 prev 是空的。
        # 但如果 prev 存在且无重叠，可能是 "Hello" -> "World" (中间断了?)
        # 简单的策略是直接追加。
        delta = cur

    if not delta:
        return base or ""
        
    # 简单的去重检查：如果 base 已经以 delta 开头（极其罕见），或者 base 结尾包含了 delta 的开头
    # 这里再做一次 base 结尾与 delta 开头的融合检查，防止微小重复
    if base:
        # 再次检查 base 的尾部和 delta 的头部
        m = min(len(base), len(delta))
        overlap = 0
        for k in range(m, 0, -1):
            if base.endswith(delta[:k]):
                overlap = k
                break
        if overlap > 0:
            delta = delta[overlap:].strip()

    if not delta:
        return base or ""

    return (base + (" " if base else "") + delta).strip()

def add_log(s):
    global logs
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="ignore")
        except Exception:
            s = s.decode("latin-1", errors="ignore")
    else:
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    ts = time.strftime("%H:%M:%S")
    logs.append(f"[{ts}] {s}")
    if len(logs) > 200:
        del logs[: len(logs) - 200]

def submit_text(text: str, dt: float | None = None, strip_ok: bool = False) -> None:
    global last_submit_lines, submissions
    if not text:
        return
    if strip_ok:
        text = re.sub(r"\s*(ok|okay)[\.!?\"]?$", "", text, flags=re.IGNORECASE).strip()
    text = _sanitize_text(text)
    
    if dt is not None:
        add_log(f"识别完成（用时{dt:.2f}秒）")
    final = text.lower()
    resp = assistant.answer(final, None)
    add_log("提交已触发")
    
    lines_for_display = []
    assistant_text = ""
    if resp:
        added = False
        if isinstance(resp, bytes):
            try:
                text_resp = resp.decode("utf-8", errors="ignore")
            except Exception:
                text_resp = resp.decode("latin-1", errors="ignore")
        else:
            text_resp = str(resp)
        text_resp = text_resp.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
        assistant_text = text_resp.strip()
        for ln in text_resp.splitlines():
            ln = ln.strip()
            if ln:
                add_log(ln)
                lines_for_display.append(ln)
                added = True
        if not added:
            add_log(text_resp)
            lines_for_display = [text_resp]
    else:
        lines_for_display = [text]
    
    try:
        last_submit_lines = lines_for_display
    except Exception:
        pass
    try:
        submissions.append({"ts": time.time(), "recognized": text, "assistant": assistant_text})
    except Exception:
        pass

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/logs")
def get_logs():
    with buffer_lock:
        live = buffer_text
    return jsonify({"logs": last_submit_lines, "live": live})

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
            block.append("----------------------------------------")
            parts.append("\n".join([x for x in block if x]))
        content = "\n\n".join(parts)
    except Exception:
        content = ""
    from flask import Response
    resp = Response(content, mimetype="text/plain; charset=utf-8")
    resp.headers["Content-Disposition"] = "attachment; filename=submissions.txt"
    return resp

@app.route("/api/audio", methods=["POST"])
def upload_audio():
    global buffer_text, speaking_active, speaking_last_ts, last_submit_ts, last_chunk_audio, last_chunk_sr, last_partial_text, last_partial_ts
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
    
    # Audio Processing
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
            
        dur = len(arr) / float(sr_rate) if arr is not None else 0.0
        rms = float(np.sqrt(np.mean(np.square(arr)))) if arr is not None and arr.size else 0.0
        
        if (not speaking_active) and (dur < MIN_SPEECH_DURATION and rms < MIN_SPEECH_RMS):
            return jsonify({"ok": True, "text": ""})

        speaking_last_ts = now_ts
        last_chunk_audio = arr.astype(np.float32)
        last_chunk_sr = int(sr_rate)
        
        if not speaking_active:
            asr_model.start_stream(sr_rate)
            speaking_active = True
            
        asr_model.push_array(arr.astype(np.float32), sr_rate)
        
        # THROTTLING: Only partial transcribe if enough time has passed
        if now_ts - last_partial_ts > PARTIAL_INTERVAL:
            _t0 = time.time()
            chunk = asr_model.partial_transcribe()
            last_partial_ts = now_ts 
            
            if chunk:
                try:
                    cur_text = (chunk.get("text") or "").strip() if isinstance(chunk, dict) else str(chunk or "").strip()
                except Exception:
                    cur_text = str(chunk or "").strip()
                try:
                    cur_seconds = float(chunk.get("seconds") or 0.0) if isinstance(chunk, dict) else None
                except Exception:
                    cur_seconds = None
                
                with buffer_lock:
                    if cur_seconds<8:
                        buffer_text = buffer_text[:-len(last_partial_text)] + cur_text
                    else:
                        buffer_text = _append_chunk(buffer_text, last_partial_text, cur_text)
                last_partial_text = cur_text
                
        b_all = (buffer_text or "").lower()
        ends_ok = re.search(r"(?:^|\s)(ok|okay)[\.!?\"]*", b_all)
        
        if ((" ok" in b_all) or (" okay" in b_all) or ends_ok) and (now_ts - last_submit_ts) > SUBMIT_COOLDOWN:
            _t1 = time.time()
            final_text = buffer_text
            speaking_active = False
            dt2 = time.time() - _t1
            submit_text(final_text, dt=dt2, strip_ok=True)
            last_submit_ts = now_ts
            buffer_text = ""
            
        return jsonify({"ok": True, "text": buffer_text})
        
    except Exception as e:
        add_log(f"ASR识别异常: {e}")
        return jsonify({"ok": True, "text": ""})

@app.route("/api/recognize", methods=["POST"])
def recognize_now():
    global last_submit_ts, last_chunk_audio, last_chunk_sr, buffer_text
    now_ts = time.time()
    with buffer_lock:
        text = buffer_text.strip()
        buffer_text = ""
    if not text:
        try:
            if last_chunk_audio is not None and getattr(last_chunk_audio, "size", 0) > 0:
                _t0 = time.time()
                asr_model.start_stream(int(last_chunk_sr or 16000))
                asr_model.push_array(last_chunk_audio.astype(np.float32), int(last_chunk_sr or 16000))
                text = asr_model.finish_stream() or ""
                dt = time.time() - _t0
                if text:
                    add_log(f"识别完成（用时{dt:.2f}秒）")
            else:
                text = ""
        except Exception as e:
            add_log(f"ASR识别异常: {e}")
            text = ""
        if not text:
            return jsonify({"ok": True, "text": ""})
    submit_text(text)
    last_submit_ts = now_ts
    return jsonify({"ok": True, "text": text})

def run():
    port = int(os.getenv("PORT", "5010"))
    # Warmup
    try:
        _warm = np.zeros(16000, dtype=np.float32)
        asr_model.start_stream(16000)
        asr_model.push_array(_warm, 16000)
        asr_model.finish_stream()
    except Exception:
        pass
        
    cert = os.getenv("TLS_CERT", "server.crt")
    key = os.getenv("TLS_KEY", "server.key")
    try:
        if os.path.exists(cert) and os.path.exists(key):
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            app.run(host="0.0.0.0", port=port, ssl_context=context)
        else:
            app.run(host="0.0.0.0", port=port)
    except OSError:
        alt = 5001 if port == 5000 else 8000
        app.run(host="0.0.0.0", port=alt)
    except Exception as e:
        add_log(f"服务启动异常: {e}")

if __name__ == "__main__":
    run()