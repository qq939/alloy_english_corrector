import base64
from threading import Lock, Thread
import openai
import json
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI
from speech_recognition import UnknownValueError

load_dotenv()


class WebcamStream:
    def __init__(self):
        import cv2 as _cv2
        self._cv2 = _cv2
        self.stream = self._cv2.VideoCapture(index=0)
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
            _, buffer = self._cv2.imencode(".jpeg", frame)
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
        self.memeory_max = 24
        self.word_n = 5        # 重点词数

        self.logger = logging.getLogger(__name__)
        self.logger.level = logging.INFO
        logfile = "assistant.log"
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

    def deduplicate_prompt(self, input):
        # 去重连续重复的字符
        import re
        input["prompt"] = re.sub(r'(.{2,}?)\1{2,}', r'\1', input["prompt"])

        return input

    def deduplicate_chat_history(self, input):
        # 去重连续重复的字符
        # 只取最近的100条记录，避免Token溢出
        messages = self.chat_message_history.messages
        if len(messages) > self.memeory_max:
            input["chat_history"] = messages[-self.memeory_max:]
        else:
            input["chat_history"] = messages

        return input

    def print_llm_output(self, input_data):
        output = ""
        try:
            """从第一个模型输出中提取Replacement words并更新词频"""
            self.logger.info(f"第一个模型输出：{input_data['llm_output'].content}")
            output = json.dumps({
            "llm_output": input_data["llm_output"].content,  # 保留第一个模型的原始输出
            "llm_output1": input_data["llm_output"].content,
            "word_n": str(self.word_n),
        })
        except Exception as e:
            self.logger.error(f"Error in 第一个模型输出: {e}")
        # 返回包含当前替换词和历史词频的字典
        return output
    def extract_replacement_words(self, input_data):
        words = []
        try:
            """从第二个模型输出中提取Replacement words并更新词频"""
            self.logger.info(f"第二个模型输出：{input_data['llm_output']}")
            lines = input_data["llm_output"].strip().split("\n")

            for line in lines:
                if "Replacement*words:" not in line:
                    words = []
                    continue
                words_part = line.split("Replacement words:")[-1].strip()
                # 第二步：按逗号分割为列表，去除空值和空格
                words = [word.strip() for word in words_part.split(",") if word.strip()]
                break
        except Exception as e:
            self.logger.error(f"Error in 第二个模型输出: {e}")
        # 返回包含当前替换词和历史词频的字典
        return json.dumps({
            "current_replacement_words": " ,".join(words),
            "llm_output": input_data["llm_output"],
            "llm_output1": input_data["llm_output1"],
        })

    def renew_word_frequency(self, input_data):
        self.logger.info(f"第三个模型的输出结果：{input_data['llm_output3']}")
        words = input_data["current_replacement_words"]
        # 从chat history解析最新的词频统计
        latest_word_frequency = {}
        for message in reversed(self.chat_message_history.messages):
            if message.type == "system":
                match = message.content.split("[Statistics] Renewed word frequency: ")[-1].strip()

                try:
                    latest_word_frequency = json.loads(match)
                except (json.JSONDecodeError, TypeError):
                    self.logger.error(f"Failed to parse JSON: {match}")
                break

        renewed_word_frequency = latest_word_frequency.copy()
        for word in words:
            if word in renewed_word_frequency:
                renewed_word_frequency[word] = renewed_word_frequency[word] + 1
            else:
                renewed_word_frequency[word] = 1

        self.chat_message_history.add_message(SystemMessage(content=f"[Statistics] Renewed word frequency: {json.dumps(renewed_word_frequency)}", additional_kwargs={"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}))

        return "\n".join([
            str(input_data.get('llm_output1', '')),
            str(input_data.get('llm_output', '')),
            str(input_data.get('llm_output3', '')),
        ]).strip()

    def answer(self, prompt, image):
        if not prompt:
            return

        self.logger.info(f"用户输入:{prompt}")


        

        response = self.chain.invoke(
            # {"prompt": prompt, "image_base64": image.decode()},
            {"prompt": prompt},
            config={"configurable": {"session_id": "unused"}},
        ).strip()
        return response

    def _tts(self, response):
        from pyaudio import PyAudio, paInt16
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
        You are an English teacher. Always respond in ENGLISH ONLY.
        If the user's sentence is not English, translate it to English first, then provide the corrected English sentence.
        Remove interjections and repetitive parts if necessary; fix mispronunciations (e.g., "walking" vs "working").
        Output format (strict), Align with colons:
        Original****sentence: <English sentence>
        Replacement*sentence: <Corrected English sentence>
        ReplacementSentence*and*ChineseTranslation: <Format ： One English sentence ： one Chinese sentence>
        ----------------------------------------
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
        You receive the English Origin sentence and the English Replacement sentence.
        Extract better version words (replaced words) from the Replacement sentence.
        Output rules:
        - Use English spelling for the word list (no quotes or brackets)
        - Provide Chinese explanation after the word list
        - If no word replaced, return an empty list
        - Drop words shorter than 6 chars
        - Drop words not nouns/verbs/adjectives, or overly basic beginner words
        - Keep less than {word_n} words in the list, if more than {word_n} words, keep the longest or noun words
        Format,Align with colons:
        ----------------------------------------
        Replacement*words: word1, word2, word3
        *中***文***解***释*: 解释1, 解释2, 解释3
        ----------------------------------------
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
        请输出英文回答。
        你是一名数据分析师，需查找给出的英文词汇是否存在于聊天记录的系统信息中。
        如果给出的英文词汇为空，无需给出任何回答。
        如果给出的英文词汇出现在聊天记录的系统信息中，且出现频次大于 2，请告知该词汇在'Renewed word frequency'中的出现次数、时间、以及系统信息所对应的人工提问的内容。
        如果给出的英文词汇未出现在聊天记录的系统信息中，则无需给出任何回答。
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
                        {"type": "text", "text": "英文词汇: {current_replacement_words}"},
                    ],
                ),
            ]
        )


        chain = RunnablePassthrough.assign(llm_output = RunnableLambda(self.deduplicate_prompt) | prompt_template | model) | RunnableLambda(self.print_llm_output) | JsonOutputParser()
        chain2 = chain | RunnablePassthrough.assign(llm_output = prompt_template2 | model | StrOutputParser()) | RunnableLambda(self.extract_replacement_words) | JsonOutputParser()
        chain3 = chain2 | RunnableLambda(self.deduplicate_chat_history) | RunnablePassthrough.assign(llm_output3 = prompt_template3 | model | StrOutputParser()) | RunnableLambda(self.renew_word_frequency) | StrOutputParser()

        
        return RunnableWithMessageHistory(
            chain3,
            lambda _: self.chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = None

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
# model = ChatOpenAI(model="qwen-vl-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",api_key=os.getenv("ALIYUN_API_KEY"))
# model = ChatOpenAI(model="gpt-4o-mini")
# Note: Pro/deepseek-ai/DeepSeek-V3.2 requires paid balance on SiliconFlow.
# Switching to a free model (THUDM/glm-4-9b-chat) for now.
model = ChatOpenAI(model="THUDM/glm-4-9b-chat", base_url="https://api.siliconflow.cn/v1", api_key=os.getenv("SILICONFLOW_API_KEY"))
# model = ChatOpenAI(model="Pro/deepseek-ai/DeepSeek-V3.2", base_url="https://api.siliconflow.cn/v1", api_key=os.getenv("SILICONFLOW_API_KEY"))

assistant = Assistant(model)

if __name__ == "__main__":
    import cv2
    from speech_recognition import Microphone, Recognizer
    webcam_stream = WebcamStream().start()
    def audio_callback(recognizer, audio):
        try:
            prompt = recognizer.recognize_whisper(audio, model="base", language="english")
            print(f"prompt:{prompt}")
            assistant.answer(prompt, None)
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
