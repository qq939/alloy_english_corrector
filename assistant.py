from ast import main
import base64
from threading import Lock, Thread
import cv2
from langchain_core.tools import structured
import openai
import json
import os
import logging
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()
logger = logging.getLogger(__name__)
logfile = "assistant.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
        self.chat_message_history = FileChatMessageHistory(file_path="lanchain_history.json")

    def deduplicate_prompt(self, input):
        # 去重连续重复的字符
        import re
        input["prompt"] = re.sub(r'(.{2,}?)\1{2,}', r'\1', input["prompt"])

        return input

    def deduplicate_chat_history(self, input):
        # 去重连续重复的字符
        input["chat_history"] = self.chat_message_history.messages

        return input

    def print_llm_output(self, input_data):

        """从第一个模型输出中提取Replacement words并更新词频"""
        logger.info(f"{input_data['llm_output'].content}")
        # 返回包含当前替换词和历史词频的字典
        return json.dumps({
            "llm_output": input_data["llm_output"].content,  # 保留第一个模型的原始输出
            "llm_output1": input_data["llm_output"].content,
        })
    def extract_replacement_words(self, input_data):

        """从第二个模型输出中提取Replacement words并更新词频"""
        logger.info(f"{input_data['llm_output']}")
        lines = input_data["llm_output"].strip().split("\n")
        for line in lines:
            if "Replacement words:" not in line:
                words = []
                continue
            words_part = line.split("Replacement words:")[-1].strip()
            # 第二步：按逗号分割为列表，去除空值和空格
            words = [word.strip() for word in words_part.split(",") if word.strip()]
            break
        # 返回包含当前替换词和历史词频的字典
        return json.dumps({
            "current_replacement_words": words,
            "llm_output": input_data["llm_output"],
            "llm_output1": input_data["llm_output1"],
        })

    def renew_word_frequency(self, input_data):
        logger.info(f"{input_data['llm_output']}")
        words = input_data["current_replacement_words"]
        # 从chat history解析最新的词频统计
        latest_word_frequency = {}
        for message in reversed(self.chat_message_history.messages):
            if message.type == "system":
                match = message.content.split("[Statistics] Renewed word frequency: ")[-1].strip()
                try:
                    latest_word_frequency = json.loads(match)
                except (json.JSONDecodeError, TypeError):
                    logger.error(f"Failed to parse JSON: {match}")
                break

        renewed_word_frequency = latest_word_frequency.copy()
        for word in words:
            if word in renewed_word_frequency:
                renewed_word_frequency[word] = renewed_word_frequency[word] + 1
            else:
                renewed_word_frequency[word] = 1

        self.chat_message_history.add_message(SystemMessage(content=f"[Statistics] Renewed word frequency: {json.dumps(renewed_word_frequency)}"))

        # 返回包含当前替换词和历史词频的字典
        return input_data['llm_output1']

    def answer(self, prompt, image):
        if not prompt:
            return


        

        response = self.chain.invoke(
            # {"prompt": prompt, "image_base64": image.decode()},
            {"prompt": prompt},
            config={"configurable": {"session_id": "unused"}},
        ).strip()


        # logger.info(f"\nRepeat offend:\n {response}")

        # if response:
        #     self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT =\
        """
        You are an English teacher having a conversation with your student, aiming to correct their incorrect usage of words. 
        Student may use wrong words or phrases, and you need to tell them how to replace their original expressions with authentic English. 
        You need to correct semantic errors caused by mispronunciation, such as pronouncing "walking" as "working". Note that the student's original utterance can have interjections and repetitive parts removed.
        Give concise answers, with the format output: "\nOriginal sentence: xxx\nReplacement sentence: xxx"
        """
        # 英语老师
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        # {
                        #     "type": "image_url",
                        #     "image_url": "data:image/jpeg;base64,{image_base64}",
                        # },
                    ],
                ),
            ]
        )

        SYSTEM_PROMPT2 =\
        """
        You are recieveing the Origin sentence and the Replacement sentence, and you need to find better version words (replaced words) in the Replacement sentence. 
        Output the replaced words, no parentheses or brackets, no single quotes or double quotes. 
        Give concise answers, with the format output: "\nReplacement words: xxx, xxx, xxx\n中文解释: xxx, xxx, xxx"
        If no word replaced, just give the answer: "\nReplacement words: "
        If the replaced word is less than 6 characters, drop that word.
        If the replaced word is not a noun or a verb, drop that word.
        If the replaced word is a simple word that non-native speakers learn at the beginner level, drop that word.
        """
        
        # 词汇
        prompt_template2 = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT2),
                (
                    "human",
                    [
                        {"type": "text", "text": "{llm_output}"},
                    ],
                ),
            ]
        )

        SYSTEM_PROMPT3 =\
        """
        You are a data analyst try to find some word is or is not in the system message of chat_history.
        If current_replacement_words is empty, just don't give any answers.
        If the word in current_replacement_words is already in system message and the word frequency is greater than 2, just tell the student how many times does the word shows in the 'Renewed word frequency' and what is the relevant human prompt. 
        If not, just don't give any answers.
        Give the concise answers.
        """
        
        # 统计专家
        prompt_template3 = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT3),
                (
                    "human",
                    # MessagesPlaceholder(variable_name="chat_history"),
                    [
                        {"type": "text", "text": "chat_history: {chat_history}"},
                        {"type": "text", "text": "current_replacement_words: {current_replacement_words}"},
                    ],
                ),
            ]
        )


        chain = RunnablePassthrough.assign(llm_output = RunnableLambda(self.deduplicate_prompt) | prompt_template | model) | RunnableLambda(self.print_llm_output) | JsonOutputParser()
        chain2 = chain | RunnablePassthrough.assign(llm_output = prompt_template2 | model | StrOutputParser()) | RunnableLambda(self.extract_replacement_words) | JsonOutputParser()
        chain3 = chain2 | RunnableLambda(self.deduplicate_chat_history) | RunnablePassthrough.assign(llm_output =prompt_template3 | model | StrOutputParser()) | RunnableLambda(self.renew_word_frequency) | StrOutputParser()

        
        return RunnableWithMessageHistory(
            chain3,
            lambda _: self.chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = WebcamStream().start()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
# model = ChatOpenAI(model="qwen-vl-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",api_key=os.getenv("ALIYUN_API_KEY"))
# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1",api_key=os.getenv("DEEPSEEK_API_KEY"))

assistant = Assistant(model)

if __name__ == "__main__":
    def audio_callback(recognizer, audio):
        try:
            prompt = recognizer.recognize_whisper(audio, model="base", language="english")
            assistant.answer(prompt, webcam_stream.read(encode=True))

        except UnknownValueError:
            print("There was an error processing the audio.")


    recognizer = Recognizer()
    microphone = Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    while True:
        cv2.imshow("webcam", webcam_stream.read())
        if cv2.waitKey(1) in [27, ord("q")]:
            break

    webcam_stream.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)
