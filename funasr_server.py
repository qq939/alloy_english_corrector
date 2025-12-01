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
from funasr_model import FunASRModel  # 引入funASR模型封装
from assistant import assistant  # 文本助理
import re  # 文本正则

app = Flask(__name__, static_folder="static", template_folder="templates")  # Flask应用
app.config["JSON_AS_ASCII"] = False  # 返回JSON允许中文

asr_model = FunASRModel("paraformer-zh")  # 初始化funASR模型

buffer_text = ""  # 缓冲文本
buffer_lock = threading.Lock()  # 缓冲锁
speaking_active = False  # 说话状态
speaking_last_ts = 0.0  # 最近说话时间戳
MIN_SPEECH_RMS = 0.08  # 最低能量阈值
MIN_SPEECH_DURATION = 0.25  # 最短时长阈值

logs = []  # 日志列表
log_once_audio = False  # 首次音频日志标记
log_once_lock = threading.Lock()  # 日志一次性锁
last_submit_ts = 0.0  # 上次提交时间
SUBMIT_COOLDOWN = 2.0  # 提交冷却时间

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

@app.route("/")  # 首页
def index():
    return render_template("index.html")  # 返回页面

@app.route("/api/logs")  # 获取日志
def get_logs():
    return jsonify({"logs": logs[-100:]})  # 返回最近100条？？？

@app.route("/api/audio", methods=["POST"])  # 音频上传并流式识别（按片返回文本）
def upload_audio():
    global buffer_text, speaking_active, speaking_last_ts, last_submit_ts, recognizer  # 使用全局状态
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
        speaking_active = True  # 标记说话态
        speaking_last_ts = now_ts  # 更新时间戳
        _t0 = time.time()  # 开始计时
        text_chunk = asr_model.transcribe_array(arr.astype(np.float32), sr_rate)  # 直接对当前片段做识别
        dt = time.time() - _t0  # 识别耗时
        if text_chunk:
            with buffer_lock:
                buffer_text = (buffer_text + " " + text_chunk).strip()  # 累积流式识别文本
            b_all = buffer_text.lower()
            ends_ok = re.search(r"(?:^|\s)(ok|okay)[\.!?\"]?$", b_all)  # 末尾触发词检测
            if ((" ok" in b_all) or (" okay" in b_all) or ends_ok) and (now_ts - last_submit_ts) > SUBMIT_COOLDOWN:
                add_log(f"识别完成（用时{dt:.2f}秒）")  # 记录耗时
                final = re.sub(r"\s*(ok|okay)[\.!?\"]?$", "", b_all, flags=re.IGNORECASE).strip()  # 去除触发词
                resp = assistant.answer(final, None)  # 调用助理
                add_log("提交已触发")  # 记录提交
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
                    buffer_text = ""  # 清空文本缓冲
                last_submit_ts = now_ts
        return jsonify({"ok": True, "text": text_chunk or ""})
    except Exception as e:
        add_log(f"ASR识别异常: {e}")  # 异常日志
        return jsonify({"ok": True, "text": ""})  # 返回空
    return jsonify({"ok": True, "text": ""})  # 默认空返回

@app.route("/api/recognize", methods=["POST"])  # 点击按钮触发以累计文本为准的提交
def recognize_now():
    global last_submit_ts  # 使用全局提交时间
    now_ts = time.time()  # 当前时间
    with buffer_lock:
        text = buffer_text.strip()  # 直接使用累计文本
        buffer_text = ""  # 清空缓冲
    if not text:
        return jsonify({"ok": True, "text": ""})
    text = _sanitize_text(text)
    add_log("识别完成")
    final = text.lower()  # 转小写
    resp = assistant.answer(final, None)  # 助理处理
    add_log("提交已触发")  # 记录提交
    if resp:
        added = False
        if isinstance(resp, bytes):
            try:
                text_resp = resp.decode("utf-8", errors="ignore")  # UTF-8
            except Exception:
                text_resp = resp.decode("latin-1", errors="ignore")  # 回退
        else:
            text_resp = str(resp)  # 字符串化
        text_resp = text_resp.replace("\r\n", "\n").replace("\r", "\n")  # 规范换行
        text_resp = text_resp.replace("\\n", "\n")  # 还原转义
        for ln in text_resp.splitlines():  # 分行日志
            ln = ln.strip()
            if ln:
                add_log(ln)
                added = True
        if not added:
            add_log(text_resp)  # 原样写入
    with buffer_lock:
        buffer_text = ""  # 清空缓冲
    last_submit_ts = now_ts  # 更新提交时间
    return jsonify({"ok": True, "text": text})  # 返回文本

def run():  # 启动服务器
    port = int(os.getenv("PORT", "5010"))  # 端口
    try:
        _warm = np.zeros(16000, dtype=np.float32)  # 预热数组
        try:
            asr_model.transcribe_array(_warm, 16000)  # 模型预热
        except Exception:
            pass  # 失败忽略
    except Exception:
        pass  # 环境异常忽略
    try:
        app.run(host="0.0.0.0", port=port)  # 启动
    except OSError:
        alt = 5001 if port == 5000 else 8000  # 备用端口
        app.run(host="0.0.0.0", port=alt)  # 启动备用端口

if __name__ == "__main__":  # 主入口
    run()  # 运行
