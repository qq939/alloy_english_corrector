# Alloy English Corrector

版本: 1.0.0

项目用于实时语音识别并将口语内容纠正为更地道的英语，支持在句末说“ok/okay”触发提交完整文本到对话模型。当前强制使用英文识别（language="en"）。

## 快速开始

- 创建并激活虚拟环境后安装依赖：
- 运行语音识别：
  ```
  /Users/jimjiang/Downloads/alloy/.venv/bin/python /Users/jimjiang/Downloads/alloy/whisper_model.py
  ```

## 关键文件

- `whisper_model.py`：实时麦克风监听、识别累积、句尾“ok/okay”触发；识别语言固定为英文。
- `assistant.py`：对话链路与摄像头图像流读取。
- `requirements.txt`：依赖清单（由虚拟环境 pip freeze 生成）。

## 注意

- 如识别在噪声环境下不稳定，可在 `whisper_model.py` 中调整 `pause_threshold` 与 `non_speaking_duration`。