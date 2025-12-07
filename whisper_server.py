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
last_committed_ts = 0.0 # Timestamp of the last committed word
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
    智能拼接：基于单词粒度寻找最大重叠，容忍 cur 前缀的微小变化 (Fuzzy Overlap)，
    并进行重复短语去除。
    """
    cur = (cur_text or "").strip()
    prev = (prev_text or "").strip()
    
    if not cur:
        return base or ""
    if not prev:
        return (base + (" " if base else "") + cur).strip()

    # --- 1. 单词粒度拆分 ---
    def tokenize(text):
        tokens = []
        for m in re.finditer(r"\b[\w']+\b", text):
            tokens.append((m.group(0).lower(), m.start(), m.end()))
        return tokens

    prev_tokens = tokenize(prev)
    cur_tokens = tokenize(cur)
    
    # --- 2. 寻找最大重叠 (Fuzzy Search) ---
    n_prev = len(prev_tokens)
    n_cur = len(cur_tokens)
    
    # 我们希望找到 prev 的后缀 在 cur 的前缀中的位置
    # 允许 cur 开头有一些 token 不匹配 (skip)，这是为了处理 Whisper 输出的不稳定性
    
    best_overlap_len = 0
    best_cur_start_idx = 0 
    
    # 限制搜索范围：允许 cur 开头跳过最多 15 个词来寻找匹配
    search_window_size = 15 
    
    # 从最长可能重叠开始尝试
    max_search = min(n_prev, n_cur)
    
    for k in range(max_search, 0, -1):
        # 目标：prev 的最后 k 个词
        target_tokens = [t[0] for t in prev_tokens[n_prev-k:]]
        
        # 在 cur 的前 (k + search_window_size) 个词中寻找 target
        found = False
        # 尝试在 cur[0] ... cur[search_window] 开始匹配
        # 确保索引不越界
        limit = min(search_window_size, n_cur - k + 1)
        for start_idx in range(limit):
            # 比较 cur[start_idx : start_idx+k]
            sub_cur = [t[0] for t in cur_tokens[start_idx : start_idx+k]]
            if sub_cur == target_tokens:
                best_overlap_len = k
                best_cur_start_idx = start_idx
                found = True
                break
        if found:
            break
            
    # --- 3. 计算 Delta ---
    if best_overlap_len > 0:
        # 重叠部分在 cur 中是 cur_tokens[best_cur_start_idx ... + len]
        # delta 从该重叠部分的结束位置开始
        last_match_token = cur_tokens[best_cur_start_idx + best_overlap_len - 1]
        split_pos = last_match_token[2]
        delta = cur[split_pos:].strip()
    else:
        # 无重叠，默认直接拼接
        delta = cur

    if not delta:
        return base or ""

    # 二次检查：简单的字符串重叠（防止单词切分导致的标点遗漏）
    if base:
        m = min(len(base), len(delta))
        for k in range(m, 0, -1):
            if base.endswith(delta[:k]):
                delta = delta[k:].strip()
                break
    
    if not delta:
        return base or ""

    candidate = (base + (" " if base else "") + delta).strip()
    
    # --- 4. 重复短语检测与去除 ---
    # 暴力检查末尾重复: "Phrase A. Phrase A." -> "Phrase A."
    
    check_str = candidate[-min(len(candidate), 300):] # 只检查末尾 300 字符
    n_check = len(check_str)
    
    # 从最大可能的重复长度开始
    for length in range(n_check // 2, 4, -1): # 最小重复长度 5 chars
        suffix = check_str[-length:]
        # 前面紧挨着是否也是 suffix
        prev_segment_end = n_check - length
        prev_segment_start = n_check - 2 * length
        
        if prev_segment_start >= 0:
            prev_segment = check_str[prev_segment_start : prev_segment_end]
            # 比较时忽略首尾空白和标点
            if suffix.strip().lower() == prev_segment.strip().lower():
                 # 发现重复，去掉后缀
                 candidate = candidate[:-length].strip()
                 break
                 
    return candidate

def _append_chunk_timestamp(base_text: str, committed_ts: float, result: dict) -> tuple[str, float]:
    """
    基于时间戳的拼接：只追加时间戳在 last_committed_ts 之后的单词。
    返回 (updated_text, new_committed_ts)
    """
    words = result.get("words", [])
    if not words:
        return base_text, committed_ts
        
    stream_seconds = result.get("stream_seconds", 0.0)
    window_seconds = result.get("window_seconds", 0.0)
    
    # 当前窗口在全局流中的起始时间
    window_start_global = max(0.0, stream_seconds - window_seconds)
    
    # 忽略窗口末尾不稳定的单词 (Right-Edge Masking)
    # 比如忽略最后 0.8 秒的内容，等待下一次窗口滑动变得稳定后再提交
    SAFETY_MARGIN = 0.8
    
    new_text_parts = []
    max_ts = committed_ts
    
    word_debug_info = []

    for w in words:
        # word['start'] 是相对于窗口起始的
        w_start_global = window_start_global + w.get("start", 0.0)
        w_end_global = window_start_global + w.get("end", 0.0)
        w_end_in_window = w.get("end", 0.0)
        
        word_text = w.get("word", "").strip()
        
        # 记录调试信息
        word_debug_info.append(f"{word_text}({w_start_global:.2f}-{w_end_global:.2f})")

        # 检查是否过于靠近窗口末尾（不稳定）
        if w_end_in_window > window_seconds - SAFETY_MARGIN:
             continue

        # 只有当单词的开始时间晚于已提交的时间戳（加一点缓冲防止临界抖动）
        if w_start_global > committed_ts + 0.05: 
            text = word_text
            if text:
                new_text_parts.append(text)
                max_ts = max(max_ts, w_end_global)
    
    if word_debug_info:
        app.logger.info(f"CommittedTS: {committed_ts:.2f}, WindowStart: {window_start_global:.2f}, Words: {', '.join(word_debug_info)}")

    if new_text_parts:
        # 拼接
        delta = " ".join(new_text_parts)
        # 处理标点符号粘连 (Whisper word通常带前导空格，但也可能不带)
        # 简单处理：如果 base 不为空，加空格
        if base_text:
             base_text = (base_text + " " + delta).strip()
        else:
             base_text = delta
             
        # 简单的去重后处理（防止极少数时间戳重叠导致的单词重复）
        base_text = re.sub(r"(\b\w+\b)(?:\s+\1\b){1,}", r"\1", base_text, flags=re.I)
        
        # --- 重复短语检测与去除 ---
        # 暴力检查末尾重复: "Phrase A. Phrase A." -> "Phrase A."
        check_str = base_text[-min(len(base_text), 300):] # 只检查末尾 300 字符
        n_check = len(check_str)
        
        # 从最大可能的重复长度开始
        for length in range(n_check // 2, 4, -1): # 最小重复长度 5 chars
            suffix = check_str[-length:]
            # 前面紧挨着是否也是 suffix
            prev_segment_end = n_check - length
            prev_segment_start = n_check - 2 * length
            
            if prev_segment_start >= 0:
                prev_segment = check_str[prev_segment_start : prev_segment_end]
                # 比较时忽略首尾空白和标点
                if suffix.strip().lower() == prev_segment.strip().lower():
                     # 发现重复，去掉后缀
                     base_text = base_text[:-length].strip()
                     break
        
    return base_text, max_ts

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
    global buffer_text, speaking_active, speaking_last_ts, last_submit_ts, last_chunk_audio, last_chunk_sr, last_partial_text, last_partial_ts, last_committed_ts
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
            last_committed_ts = 0.0
            
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
                    # Prefer window_seconds if available, else fallback to seconds
                    val = chunk.get("window_seconds") if isinstance(chunk, dict) else None
                    if val is None and isinstance(chunk, dict):
                         val = chunk.get("seconds")
                    cur_seconds = float(val) if val is not None else 0.0
                except Exception:
                    cur_seconds = 0.0
                
                with buffer_lock:
                    app.logger.info(f"Partial Transcribe: {cur_text} (seconds: {cur_seconds})")
                    # 优先使用时间戳拼接
                    if isinstance(chunk, dict) and "words" in chunk and "stream_seconds" in chunk:
                         buffer_text, last_committed_ts = _append_chunk_timestamp(buffer_text, last_committed_ts, chunk)
                    else:
                        app.logger.info(f"Buffer Text: {buffer_text}")
                        if cur_seconds is not None and cur_seconds<8:
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