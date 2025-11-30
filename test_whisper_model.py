import time
import speech_recognition as sr
from whisper_model import WhisperModel

def main():
    wm = WhisperModel("small")
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
    def cb(rec, audio):
        try:
            wav = audio.get_wav_data(convert_rate=16000)
            text = wm.transcribe_wav(wav)
            if text:
                print(text)
        except Exception as e:
            print(str(e))
    stop = r.listen_in_background(mic, cb)
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop(wait_for_stop=False)

if __name__ == "__main__":
    main()