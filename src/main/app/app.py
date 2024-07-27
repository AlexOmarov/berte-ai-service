"""
Script for executing first call to chatLLM
"""
import os

import scipy
import torch

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from business.util.ml_logger.logger import create_logger


def _generate(text: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(device)

    audio_values = model.generate(**inputs, max_new_tokens=512)
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


def _call_chat(content: str) -> BaseMessage:
    logger = create_logger("APP")
    llm = ChatOllama(model="llama3:8b", num_gpu=10000, base_url=os.environ['OLLAMA_HOST'])

    messages = [
        HumanMessage(
            content=content
        )
    ]

    res = llm.invoke(messages)
    logger.info(res)
    return res


_generate("American pop-rock")
