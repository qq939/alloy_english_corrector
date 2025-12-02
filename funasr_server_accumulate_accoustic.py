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
import speech_recognition as sr  # 轻量触发词识别
from speech_recognition import UnknownValueError  # 无法识别异常
import re  # 文本正则

app = Flask(__name__, static_folder="static", template_folder="templates")  # Flask应用
app.config["JSON_AS_ASCII"] = False  # 返回JSON允许中文

asr_model = FunASRModel("paraformer-zh")  # 初始化funASR模型

buffer_text = ""  # 缓冲文本
buffer_lock = threading.Lock()  # 缓冲锁
speaking_active = False  # 说话状态
speaking_last_ts = 0.0  # 最近说话时间戳
MIN_SPEECH_RMS = 0.008  # 最低能量阈值
MIN_SPEECH_DURATION = 0.25  # 最短时长阈值

logs = []  # 日志列表
log_once_audio = False  # 首次音频日志标记
log_once_lock = threading.Lock()  # 日志一次性锁
last_submit_ts = 0.0  # 上次提交时间
SUBMIT_COOLDOWN = 1.0  # 提交冷却时间

audio_chunks = []  # 累积音频片段数组
audio_lock = threading.Lock()  # 音频锁
recognizer = None  # 轻量触发词识别器

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

@app.route("/api/audio", methods=["POST"])  # 音频上传并流式累积
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
        if not getattr(asr_model, "_streaming", False):  # 未开始流式则启动
            asr_model.start_stream(sr_rate)
        asr_model.push_array(arr, sr_rate)  # 推入流式缓冲
        with audio_lock:
            audio_chunks.append(arr)  # 本地累积片段（用于前端按钮识别）
        speaking_active = True  # 标记说话态
        speaking_last_ts = now_ts  # 更新时间戳
        if recognizer is None:  # 初始化轻量触发词识别
            recognizer = sr.Recognizer()
        try:
            ad = sr.AudioData(pcm_bytes, sr_rate, 2)  # 构造AudioData
            phrase = recognizer.recognize_whisper(ad, model="base", language="english")  # 仅检测ok/okay
        except UnknownValueError:
            phrase = ""  # 无触发词
        b = (phrase or "").strip().lower()  # 规范化触发文本
        ends_ok = re.search(r"(?:^|\s)(ok|okay)[\.!?\"]?$", b)  # 末尾触发词检测
        if ((b in {"ok", "okay"}) or ends_ok) and (now_ts - last_submit_ts) > SUBMIT_COOLDOWN:  # 冷却后触发
            with audio_lock:
                if audio_chunks:
                    full = np.concatenate(audio_chunks).astype(np.float32)  # 拼接整段
                    audio_chunks.clear()
                else:
                    full = arr.astype(np.float32)  # 仅当前片段
            full = np.clip(full, -1.0, 1.0)  # 裁剪幅度
            full = np.ascontiguousarray(full, dtype=np.float32)  # 连续内存
            _t0 = time.time()  # 开始计时
            text = asr_model.finish_stream()  # 结束流式并识别
            dt = time.time() - _t0  # 识别耗时
            if text:
                text = _sanitize_text(text)  # 清洗文本
                add_log(f"识别完成（用时{dt:.2f}秒）")  # 记录耗时
                final = re.sub(r"\s*(ok|okay)[\.!?\"]?$", "", text.lower(), flags=re.IGNORECASE).strip()  # 去除触发词
                resp = assistant.answer(final, None)  # 调用助理
                add_log("提交已触发")  # 记录提交
                if resp:
                    added = False
                    if isinstance(resp, bytes):
                        try:
                            text_resp = resp.decode("utf-8", errors="ignore")  # 尝试UTF-8
                        except Exception:
                            text_resp = resp.decode("latin-1", errors="ignore")  # 回退Latin-1
                    else:
                        text_resp = str(resp)  # 字符串化
                    text_resp = text_resp.replace("\r\n", "\n").replace("\r", "\n")  # 规范换行
                    text_resp = text_resp.replace("\\n", "\n")  # 还原转义
                    for ln in text_resp.splitlines():  # 分行写入日志
                        ln = ln.strip()
                        if ln:
                            add_log(ln)
                            added = True
                    if not added:
                        add_log(text_resp)  # 原样写入
                with buffer_lock:
                    buffer_text = ""  # 清空缓冲
                last_submit_ts = now_ts  # 更新提交时间
            return jsonify({"ok": True, "text": text or ""})  # 返回识别文本
    except Exception as e:
        add_log(f"ASR识别异常: {e}")  # 异常日志
        return jsonify({"ok": True, "text": ""})  # 返回空
    return jsonify({"ok": True, "text": ""})  # 默认空返回

@app.route("/api/recognize", methods=["POST"])  # 点击按钮触发整段识别
def recognize_now():
    global last_submit_ts  # 使用全局提交时间
    now_ts = time.time()  # 当前时间
    with audio_lock:
        if not audio_chunks:  # 无累计片段
            return jsonify({"ok": True, "text": ""})
        full = np.concatenate(audio_chunks).astype(np.float32)  # 拼接整段
        audio_chunks.clear()  # 清空
    full = np.clip(full, -1.0, 1.0)  # 裁剪幅度
    full = np.ascontiguousarray(full, dtype=np.float32)  # 连续内存
    try:
        _t0 = time.time()  # 计时开始
        text = asr_model.finish_stream()  # 结束流式并识别
        dt = time.time() - _t0  # 用时
    except Exception as e:
        add_log(f"ASR识别异常: {e}")  # 异常日志
        return jsonify({"ok": True, "text": ""})  # 空返回
    if not text:  # 无文本
        return jsonify({"ok": True, "text": ""})
    text = _sanitize_text(text)  # 清洗文本
    add_log(f"识别完成（用时{dt:.2f}秒）")  # 记录耗时
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
