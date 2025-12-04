import io  # 字节流支持
import numpy as np  # 数值与音频数组处理
from scipy.io import wavfile  # 读取/写入 WAV
import whisper  # OpenAI Whisper 推理

"""
英文流式识别模块（Whisper Streaming）。
目标语言固定为英语（language="en"），输出也是英语（task="transcribe"）。
支持 small 与 large-v3 两种模型规格。
"""

class WhisperStreamingModel:
    """Whisper 英文流式识别模型。"""
    def __init__(self, model_size: str = "small"):
        """初始化并加载模型，model_size 可为 "small" 或 "large-v3"。"""
        self.model = whisper.load_model(model_size)  # 加载指定大小的模型
        self.language = "en"  # 固定识别目标语言为英语
        self._stream_chunks = []  # 累计的流式音频片段（16k, float32, 单声道）
        self._stream_sr = None  # 流式采样率记录（用于预处理）
        self._streaming = False  # 是否处于流式会话中

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """将输入数组归一化为 float32 [-1, 1] 区间。"""
        orig = x.dtype  # 原始数据类型
        y = x.astype(np.float32)  # 转 float32
        if orig.kind in ("i", "u"):  # 整型/无符号整型需要缩放到 [-1,1]
            scale = 32768.0 if orig.itemsize == 2 else 2147483648.0  # 16/32 位缩放因子
            y /= scale  # 归一化
        return y  # 返回 float32

    def _to_mono(self, x: np.ndarray) -> np.ndarray:
        """多通道转单声道。"""
        if getattr(x, "ndim", 1) > 1:  # 多通道
            return np.mean(x, axis=1)  # 求均值为单声道
        return x  # 已是单声道

    def _resample_16k(self, x: np.ndarray, rate: int) -> np.ndarray:
        """重采样到 16kHz。"""
        target = 16000  # 目标采样率
        if rate == target:
            return x.astype(np.float32)  # 已是 16k
        if rate % target == 0:  # 简单下采样倍数
            step = rate // target
            return x[::step].astype(np.float32)
        new_len = max(1, int(round(len(x) * target / float(rate))))  # 线性插值长度
        x_old = np.linspace(0, len(x) - 1, num=len(x))  # 原索引
        x_new = np.linspace(0, len(x) - 1, num=new_len)  # 新索引
        return np.interp(x_new, x_old, x).astype(np.float32)  # 插值

    def _prepare(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """！！！！！！！！！！！！！！！！！！！！！！！统一预处理：单声道、归一化、16k、clip、连续内存。"""
        mono = self._to_mono(audio)  # 单声道
        norm = self._normalize(mono)  # 归一化
        x = self._resample_16k(norm, int(rate))  # 重采样 16k
        x = np.clip(x.astype(np.float32), -1.0, 1.0)  # 裁剪到 [-1,1]
        return np.ascontiguousarray(x)  # 连续内存

    def start_stream(self, sample_rate: int) -> None:
        """开始一次流式会话。"""
        self._stream_chunks = []  # 清空累计片段
        self._stream_sr = int(sample_rate)  # 记录采样率
        self._streaming = True  # 标记流式中

    def push_array(self, audio: np.ndarray, sample_rate: int) -> None:
        """推入一段 float32 音频。"""
        if not self._streaming:  # 未开始会话则忽略
            return
        sr = int(sample_rate)  # 当前段采样率
        x = self._prepare(audio, sr)  # 预处理到 16k/单声道/float32
        self._stream_chunks.append(x)  # 追加到累计队列

    def push_pcm16(self, raw: bytes, sample_rate: int) -> None:
        """推入一段 PCM16 原始字节。"""
        if not self._streaming or not raw or sample_rate <= 0:  # 参数检查
            return
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)  # int16->float32
        x /= 32768.0  # 归一化到 [-1,1]
        self.push_array(x, sample_rate)  # 复用数组推入流程

    def push_wav(self, raw_wav: bytes) -> None:
        """推入一段 WAV 字节。"""
        if not self._streaming:  # 未开始会话
            return
        rate, audio = wavfile.read(io.BytesIO(raw_wav))  # 解析采样率与数组
        audio = np.asarray(audio)  # 保证为 ndarray
        if audio.dtype != np.float32:  # 非 float32 先归一化
            audio = self._normalize(audio)
        self.push_array(audio, int(rate))  # 复用数组推入流程

    def partial_transcribe(self) -> str:
        """对当前累计的音频做一次英文转写（不中断流）。"""
        if not self._stream_chunks:  # 无数据则返回空
            return ""
        full = np.concatenate(self._stream_chunks).astype(np.float32)  # 合并片段
        x = self._prepare(full, self._stream_sr or 16000)  # 预处理
        result = self.model.transcribe(x, language=self.language, task="translate")  # 英文翻译
        return (result.get("text") or "").strip()  # 返回文本

    def finish_stream(self) -> str:
        """结束会话并返回最终英文文本。"""
        if not self._streaming or not self._stream_chunks:  # 无会话或无数据
            self._streaming = False
            self._stream_chunks = []
            self._stream_sr = None
            return ""
        full = np.concatenate(self._stream_chunks).astype(np.float32)  # 合并片段
        self._stream_chunks = []  # 清空缓存
        sr = self._stream_sr or 16000  # 采样率
        x = self._prepare(full, sr)  # 预处理
        result = self.model.transcribe(x, language=self.language, task="transcribe")  # 英文转写
        self._streaming = False  # 标记结束
        self._stream_sr = None  # 清理采样率
        return (result.get("text") or "").strip()  # 返回文本

def main():
    """命令行示例：从 WAV 文件模拟流式识别，语言/输出均为英语。"""
    import argparse  # 解析命令行参数
    import sys  # 系统交互

    parser = argparse.ArgumentParser(description="Whisper English Streaming Demo")  # 构建参数解析器
    parser.add_argument("input", help="输入 WAV 文件路径")  # 必选：输入音频
    parser.add_argument("--model", dest="model", default="small", choices=["small", "large-v3"], help="模型规格：small 或 large-v3")  # 模型规格
    parser.add_argument("--chunk_ms", dest="chunk_ms", type=int, default=400, help="每次推入的片段时长（毫秒）")  # 片段长度
    parser.add_argument("--print_step", dest="print_step", type=int, default=5, help="每推入多少片段做一次 partial_transcribe 打印")  # 打印频率
    args = parser.parse_args()  # 解析参数

    try:
        with open(args.input, "rb") as f:  # 打开输入文件
            raw_wav = f.read()  # 读取字节
    except Exception as e:
        print(f"[Error] 无法读取输入文件: {e}")  # 读取失败提示
        sys.exit(1)  # 退出

    # 解析 WAV，获取采样率与数组
    rate, audio = wavfile.read(io.BytesIO(raw_wav))  # 解析 WAV 字节
    audio = np.asarray(audio)  # 保证 ndarray
    if audio.dtype.kind in ("i", "u"):  # 整型归一化
        audio = audio.astype(np.float32) / 32768.0  # [-1,1]
    else:
        audio = audio.astype(np.float32)  # 转 float32

    # 初始化流式模型
    ws = WhisperStreamingModel(args.model)  # 创建模型实例
    ws.start_stream(rate)  # 开始流式会话

    # 将整段音频切分为固定毫秒片段推入
    step = max(1, int(rate * (args.chunk_ms / 1000.0)))  # 片段长度（样本数）
    pushed = 0  # 已推入片段计数
    for i in range(0, len(audio), step):  # 逐片段遍历
        chunk = audio[i:i + step]  # 取片段数组
        if not chunk.size:  # 空片段跳过
            continue
        ws.push_array(chunk, rate)  # 推入片段
        pushed += 1  # 计数
        if pushed % args.print_step == 0:  # 到达打印步长
            interim = ws.partial_transcribe()  # 局部英文转写
            if interim:
                print(f"[Interim] {interim}")  # 打印临时结果

    # 结束并输出最终英文识别文本
    final_text = ws.finish_stream()  # 结束会话并拿最终结果
    print(f"[Final] {final_text}")  # 打印最终结果

if __name__ == "__main__":
    main()  # 运行演示
