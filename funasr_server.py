from flask import Flask, request, jsonify, render_template  # 引入Flask核心API
import os  # 操作系统环境变量
os.environ.setdefault("TQDM_DISABLE", "1")  # 关闭tqdm日志干扰
import multiprocessing as mp  # 多进程设置
try:
    mp.set_start_method("spawn")  # macOS/安全的进程启动方式
except RuntimeError:
    pass  # 已设置则忽略
import io  # 字节流处理
import threading  # 线程与锁
import time  # 时间与计时
import numpy as np  # 数值计算
from scipy.io import wavfile  # 读写WAV
from whisper_streaming import WhisperStreamingModel
from funasr_model import FunASRModel

from assistant import assistant  # 文本助理
import re  # 文本正则
import ssl
import logging



app = Flask(__name__, static_folder="static", template_folder="templates")  # Flask应用
app.config["JSON_AS_ASCII"] = False  # 返回JSON允许中文
handler = logging.FileHandler("funasr_server.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(message)s %(asctime)s"))
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)
app.logger.debug("ok")

asr_model = WhisperStreamingModel("small")
# asr_model = FunASRModel("small")


buffer_text = ""  # 缓冲文本
last_chunk_audio = None
last_chunk_sr = 16000
last_partial_text = ""
last_partial_offset = None
last_words = []
buffer_lock = threading.Lock()  # 缓冲锁
speaking_active = False  # 说话状态
speaking_last_ts = 0.0  # 最近说话时间戳
MIN_SPEECH_RMS = 0.08
MIN_SPEECH_DURATION = 0.25  # 最短时长阈值

logs = []  # 日志列表
log_once_audio = False  # 首次音频日志标记
log_once_lock = threading.Lock()  # 日志一次性锁
last_submit_ts = 0.0  # 上次提交时间
SUBMIT_COOLDOWN = 2.0

submissions = []  # 累计提交的文本内容（用于下载）
last_submit_lines = []  # 最近一次提交的展示内容（用于日志面板）

recognizer = None  # 保留占位，不再使用轻量触发词识别

def _sanitize_text(s: str) -> str:  # 文本清洗
    s = re.sub(r"\s+", " ", s).strip()  # 归并空白
    parts = re.split(r"([,\.\!])", s)  # 分割标点
    kept = []  # 保留段
    for i in range(0, len(parts), 2):  # 两步取正文与标点
        if i >= len(parts):
            break
        body = parts[i].strip()  # 正文
        if not body:
            continue
        punct = parts[i + 1] if i + 1 < len(parts) else ""  # 标点
        if re.search(r"\bthank\s+you\b", body, flags=re.I):  # 去除客套
            continue
        kept.append(body + (punct or ""))  # 合并
    s = " ".join(kept).strip()  # 拼接
    s = re.sub(r"(\b\w+\b)(?:\s+\1\b){1,}", r"\1", s, flags=re.I)  # 连续词去重
    s = re.sub(r"(\b\w+\b[\.\!\?])(?:\s+\1){1,}", r"\1", s, flags=re.I)  # 句子去重
    return s

def _append_chunk_directly(base: str, cur_text: str) -> str:
    if not cur_text:
        return base or ""
    if not base:
        return cur_text or ""
    return base + "" + cur_text


def _append_chunk_offset(base: str, prev_text: str, prev_offset, cur_text: str, cur_offset) -> str:
    app.logger.debug("cur_text:" + cur_text)
    cur = (cur_text or "").strip()
    if not cur:
        return base or ""
    eps = 1e-3
    # 优先按偏移处理：窗口回退则视为新的段，窗口前进则按前缀差异增量
    if prev_text:
        if (prev_offset is not None) and (cur_offset is not None):
            if float(cur_offset) + eps < float(prev_offset):
                tail = base[-len(cur):] if base else ""
                n = min(len(tail), len(cur))
                i = 0
                while i < n and tail[i] == cur[i]:
                    i += 1
                delta = cur[i:].strip()
                if not delta:
                    return base or ""
                return (base + (" " if base else "") + delta).strip()
            else:
                prev = (prev_text or "").strip()
                n = min(len(prev), len(cur))
                i = 0
                while i < n and prev[i] == cur[i]:
                    i += 1
                delta = cur[i:].strip()
                if not delta:
                    return base or ""
                return (base + (" " if base else "") + delta).strip()
    # 无前一段，直接追加当前段
    return (base + (" " if base else "") + cur).strip()

def add_log(s):  # 统一日志追加
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="ignore")  # 优先UTF-8
        except Exception:
            s = s.decode("latin-1", errors="ignore")  # 回退Latin-1
    else:
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")  # 规范换行
    s = s.replace("\\n", "\n")  # 还原转义换行
    ts = time.strftime("%H:%M:%S")  # 时间戳
    logs.append(f"[{ts}] {s}")  # 加入日志
    if len(logs) > 200:
        del logs[: len(logs) - 200]  # 限制长度

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
        text_resp = text_resp.replace("\r\n", "\n").replace("\r", "\n")
        text_resp = text_resp.replace("\\n", "\n")
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

@app.route("/")  # 首页
def index():
    return render_template("index.html")  # 返回页面

@app.route("/api/logs")  # 获取日志
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

@app.route("/api/audio", methods=["POST"])  # 音频上传并流式识别（按片返回文本）
def upload_audio():
    global buffer_text, speaking_active, speaking_last_ts, last_submit_ts, recognizer, last_chunk_audio, last_chunk_sr, last_partial_offset, last_partial_text # 使用全局状态
    file_raw = request.files.get("audio_raw")  # PCM16文件
    file_wav = request.files.get("audio")  # WAV文件
    if not file_raw and not file_wav:  # 无文件
        return jsonify({"ok": False, "error": "no audio"}), 400
    raw = (file_raw or file_wav).read()  # 读字节
    with log_once_lock:   # 首次日志记录锁？？
        global log_once_audio
        if not log_once_audio:
            add_log("音频片段已接收")  # 首次日志
            log_once_audio = True
    now_ts = time.time()  # 当前时间
    sr_rate = 16000  # 默认采样率
    arr = None  # 数组形式音频
    try:
        if file_raw:  # 处理PCM16
            try:
                sr_rate = int(request.form.get("sr", "16000"))  # 获取采样率
            except Exception:
                sr_rate = 16000  # 回退
            if sr_rate <= 0 or sr_rate < 8000 or sr_rate > 96000:  # 异常采样率
                return jsonify({"ok": True, "text": ""})
            if not raw:  # 空数据
                return jsonify({"ok": True, "text": ""})
            nsamples = len(raw) // 2  # 采样数，向下取整除以2，因为每个采样2字节
            min_samples = int(max(1600, sr_rate * 0.3))  # 最小长度约0.3s
            if nsamples < min_samples:  # 过短
                return jsonify({"ok": True, "text": ""})
            if len(raw) % 2 != 0:  # 奇数字节对齐
                raw = raw[: len(raw) - 1]
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0  # 转浮点归一化
            arr = x  # 片段数组
            pcm_bytes = raw  # 原始PCM字节
        else:  # 处理WAV  --->   PCM字节
            rate, audio_np = wavfile.read(io.BytesIO(raw))  # 读WAV
            try:
                sr_rate = int(rate)  # 采样率
            except Exception:
                sr_rate = 16000
            if sr_rate <= 0 or sr_rate < 8000 or sr_rate > 96000:  # 异常采样率
                return jsonify({"ok": True, "text": ""})
            if getattr(audio_np, "ndim", 1) > 1:  # 多通道转单声道
                audio_np = np.mean(audio_np, axis=1)
            if audio_np.dtype.kind in ("i", "u"):  # 整型归一化
                x = audio_np.astype(np.float32) / 32768.0
            else:
                x = audio_np.astype(np.float32)
            arr = x  # 片段数组
            y = np.clip(x, -1.0, 1.0)  # 幅度裁剪[-1,1]闭区间
            pcm_bytes = (y * 32767.0).astype(np.int16).tobytes()  # 生成PCM字节用于触发识别
        dur = len(arr) / float(sr_rate) if arr is not None else 0.0  # 片段时长
        rms = float(np.sqrt(np.mean(np.square(arr)))) if arr is not None and arr.size else 0.0  # 能量
        if (not speaking_active) and (dur < MIN_SPEECH_DURATION and rms < MIN_SPEECH_RMS):  # 过滤短/弱片段
            return jsonify({"ok": True, "text": ""})

        speaking_last_ts = now_ts  # 更新时间戳
        last_chunk_audio = arr.astype(np.float32)  # 记录最近片段供按钮兜底识别
        last_chunk_sr = int(sr_rate)
        if not speaking_active:
            asr_model.start_stream(sr_rate)
            speaking_active = True
        asr_model.push_array(arr.astype(np.float32), sr_rate)
        _t0 = time.time()
        chunk = asr_model.partial_transcribe()
        if chunk:
            try:
                cur_text = (chunk.get("text") or "").strip() if isinstance(chunk, dict) else str(chunk or "").strip()
            except Exception:
                cur_text = str(chunk or "").strip()
            try:
                cur_offset = float(chunk.get("offset") or 0.0) if isinstance(chunk, dict) else None
            except Exception:
                cur_offset = None
            app.logger.debug("Before Lock text"+cur_text)
            app.logger.debug("Before Lock offset"+str(cur_offset))
            with buffer_lock:
                # buffer_text = _append_chunk_offset(buffer_text, last_partial_text, last_partial_offset, cur_text, cur_offset)
                buffer_text = _append_chunk_directly(buffer_text,  cur_text)
            last_partial_text = cur_text
            last_partial_offset = cur_offset
        b_all = (buffer_text or "").lower()
        ends_ok = re.search(r"(?:^|\s)(ok|okay)[\.!?\"]*", b_all)
        if ((" ok" in b_all) or (" okay" in b_all) or ends_ok) and (now_ts - last_submit_ts) > SUBMIT_COOLDOWN:
            _t1 = time.time()
            res = asr_model.finish_stream()
            final_text = (res.get("text") or "").strip() if isinstance(res, dict) else str(res or "").strip()
            speaking_active = False
            dt2 = time.time() - _t1
            submit_text(final_text, dt=dt2, strip_ok=True)
            last_submit_ts = now_ts
            last_words = []
            buffer_text = ""
        return jsonify({"ok": True, "text": buffer_text})
    except Exception as e:
        add_log(f"ASR识别异常: {e}")  # 异常日志
        return jsonify({"ok": True, "text": ""})  # 返回空
    return jsonify({"ok": True, "text": ""})  # 默认空返回

@app.route("/api/recognize", methods=["POST"])  # 点击按钮触发以累计文本为准的提交
def recognize_now():
    global last_submit_ts, last_chunk_audio, last_chunk_sr, buffer_text  # 使用全局提交时间与最近片段
    now_ts = time.time()  # 当前时间
    with buffer_lock:
        text = buffer_text.strip()  # 直接使用累计文本
        buffer_text = ""  # 清空缓冲
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

def run():  # 启动服务器
    port = int(os.getenv("PORT", "5010"))  # 端口
    try:
        _warm = np.zeros(16000, dtype=np.float32)
        try:
            asr_model.start_stream(16000)
            asr_model.push_array(_warm, 16000)
            asr_model.finish_stream()
        except Exception:
            pass
    except Exception:
        pass
    cert = os.getenv("TLS_CERT", "server.crt")  # 证书路径
    key = os.getenv("TLS_KEY", "server.key")  # 私钥路径
    try:
        if os.path.exists(cert) and os.path.exists(key):  # 若证书存在，启用HTTPS
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            app.run(host="0.0.0.0", port=port, ssl_context=context)
        else:  # 否则使用HTTP
            app.run(host="0.0.0.0", port=port)
    except OSError:
        alt = 5001 if port == 5000 else 8000  # 备用端口
        app.run(host="0.0.0.0", port=alt)
    except Exception as e:
        add_log(f"服务启动异常: {e}")

if __name__ == "__main__":  # 主入口
    run()  # 运行
