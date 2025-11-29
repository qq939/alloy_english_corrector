from cv2 import log
import speech_recognition as sr
import whisper
import threading
import numpy as np
from io import BytesIO
from scipy.io import wavfile
from assistant import assistant, webcam_stream
import time
from queue import Queue, Full
import logging
log = logging.getLogger(__name__)
logfile = "wisper_logfile.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MIN_SPEECH_RMS = 0.008
MIN_SPEECH_DURATION = 0.25

# 1. 初始化Whisper模型（CPU选small，GPU改medium）
whisper_model = whisper.load_model("small")

# 2. 初始化麦克风和识别器
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# 3. 噪音校准
with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
    recognizer.dynamic_energy_threshold = True
    recognizer.dynamic_energy_ratio = 1.5
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.8

# 4. 核心回调函数（修复非法参数+保留核心逻辑）
buffer_text = ""
buffer_lock = threading.Lock()
stop_handle = None
current_lang = "en"
lang_lock = threading.Lock()
lang_last_update = 0.0
lang_queue: Queue = Queue(maxsize=3)
speaking_active = False
speaking_last_ts = 0.0

def _has_chinese(text: str) -> bool:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False

def audio_callback(recognizer, audio):
    global buffer_text, stop_handle, current_lang, lang_last_update, speaking_active, speaking_last_ts
    try:
        # 步骤1：音频格式转换（bytes → numpy数组）
        audio_data = audio.get_wav_data(convert_rate=16000)
        wav_stream = BytesIO(audio_data)
        sample_rate, audio_np = wavfile.read(wav_stream)
        
        # 格式适配Whisper要求
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)  # 多通道转单通道
        audio_np = audio_np.astype(np.float32) / 32768.0  # 归一化，整型转浮点型（-1~1）

        duration = len(audio_np) / float(sample_rate)
        rms = float(np.sqrt(np.mean(np.square(audio_np))))
        now_ts = time.time()
        if (not speaking_active) and (duration < MIN_SPEECH_DURATION and rms < MIN_SPEECH_RMS):
            return

        with lang_lock:
            use_lang = "en"
            last_ts = lang_last_update

        # 不再进行异步语言检测

        result = whisper_model.transcribe(
            audio_np,
            no_speech_threshold=0.5,
            logprob_threshold=-1.1,
            temperature=0.0,
            prompt='请识别中文、英文，包括中英混说，允许正常口语停顿，无语音时返回空，不要补充任何无关文本["Thanks for watching !","謝謝收看"]。统一输出为英文。',
            language=use_lang,
            fp16=False,
            task="transcribe",
            verbose=False,
        )

        # 步骤3：输出结果
        detected_language = result["language"]
        recognized_text = result["text"].strip()
        if not recognized_text:
            if rms >= MIN_SPEECH_RMS:
                speaking_active = True
                speaking_last_ts = now_ts
            elif speaking_active and (now_ts - speaking_last_ts) > 2.0:
                speaking_active = False
            return

        buffer_lock.acquire()
        try:
            global buffer_text, stop_handle
            buffer_text = (buffer_text + " " + recognized_text).strip()
            b = buffer_text.lower()
        finally:
            buffer_lock.release()

        speaking_active = True
        speaking_last_ts = now_ts

        log.info(f"自动检测语言：{detected_language}")
        log.info(f"当前累计识别：{buffer_text}\n")

        if " ok" in b:
            # 去除结尾的触发词

            b = b.replace("okay", "").replace("ok", "").strip()

            # if stop_handle:
            #     stop_handle(wait_for_stop=False)

            assistant.answer(b, webcam_stream.read(encode=True))
            buffer_text = ""
        
    except Exception as e:
        log.warning(f"识别异常：{str(e)}\n")

# 5. 启动实时监听
stop_handle = recognizer.listen_in_background(
    microphone,
    audio_callback,
    phrase_time_limit=60,
)

# 维持程序运行
try:
    while True:
        threading.Event().wait()
except KeyboardInterrupt:
    if stop_handle:
        stop_handle(wait_for_stop=False)
    log.warning("监听已停止")
