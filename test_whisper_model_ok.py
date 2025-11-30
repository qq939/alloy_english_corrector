import time  # 用于定时与时间戳
import threading  # 线程与锁：在回调里安全累积音频
import numpy as np  # 数值计算，用于拼接整段音频
from io import BytesIO  # 将WAV字节读为文件流
from scipy.io import wavfile  # 解析WAV为采样率与数组
import speech_recognition as sr  # 麦克风采集与Whisper触发词探测
from speech_recognition import UnknownValueError  # 探测失败异常
from whisper_model import WhisperModel  # 我们封装的Whisper类（最终整段识别）

MIN_SPEECH_RMS = 0.01  # 最低能量阈值：过滤极低能量片段
MIN_SPEECH_DURATION = 0.3  # 最短时长阈值：过滤过短片段

def on_ok(text):
    # 触发词“ok/okay”出现后，对整段音频识别并回调输出
    print("[OK]", text)

def main():
    wm = WhisperModel("small")  # 最终整段识别使用small+translate
    r = sr.Recognizer()  # SR用于回调与触发词“轻量探测”
    mic = sr.Microphone()  # 默认麦克风设备
    with mic as source:
        # 噪声自适应与端点参数，提升触发准确性
        r.adjust_for_ambient_noise(source, duration=1) # 初始1秒内自适应噪声
        r.dynamic_energy_threshold = True  # 动态阈值：过滤背景噪声
        r.dynamic_energy_ratio = 1.5  # 动态频阈值：1.5倍于静态阈值
        r.pause_threshold = 0.8  # 暂停阈值：0.8秒无声音为停顿
        r.non_speaking_duration = 0.8  # 非语音持续时间：0.8秒无声音为非语音
    speaking_active = False  # 是否处于说话态
    speaking_last_ts = 0.0  # 最近说话时间戳
    audio_chunks = []  # 累积的整段音频（Float32）
    lock = threading.Lock()  # 保护累积数组的线程安全
    def cb(rec, audio):  # 麦克风回调：每段到来时尝试触发检测与累积
        nonlocal speaking_active, speaking_last_ts, audio_chunks
        try:
            wav = audio.get_wav_data(convert_rate=16000)  # 统一转为16k WAV字节
            sr_rate, arr = wavfile.read(BytesIO(wav))  # 解析出采样率与数组
            if getattr(arr, "ndim", 1) > 1:
                arr = np.mean(arr, axis=1)  # 多通道转单声道
            arr = arr.astype(np.float32) / 32768.0  # 归一化到[-1,1]
            dur = len(arr) / float(sr_rate)  # 片段时长（秒）
            rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0  # 能量
            now_ts = time.time()
            if (not speaking_active) and (dur < MIN_SPEECH_DURATION and rms < MIN_SPEECH_RMS):
                # 该片段过短且能量低，不累积也不触发
                return
            with lock:
                audio_chunks.append(arr)  # 累积片段供“整段识别”使用
            speaking_active = True
            speaking_last_ts = now_ts
            try:
                # 仅用SR的“base模型”做轻量触发词探测（快）
                phrase = rec.recognize_whisper(audio, model="base", language="english")
            except UnknownValueError:
                phrase = ""
            b = (phrase or "").strip().lower()
            if (b in {"ok", "okay"}) or (" ok" in b) or (" okay" in b):
                with lock:
                    if not audio_chunks:  # 无累积音频，不触发，为啥没有累积到？
                        return
                    # 拼接整段音频并清空缓冲，不写就是第0维
                    full = np.concatenate(audio_chunks).astype(np.float32)
                    audio_chunks = []
                # 最终一次性整段识别（translate任务）
                text = wm.transcribe_array(full, 16000)
                on_ok(text)
                speaking_active = False
                speaking_last_ts = now_ts
        except Exception as e:
            print(str(e))
    stop = r.listen_in_background(mic, cb, phrase_time_limit=60)
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop(wait_for_stop=False)

if __name__ == "__main__":
    main()