"""
Script for executing first call to chatLLM
"""
import os

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage

from business.util.ml_logger.logger import create_logger


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
