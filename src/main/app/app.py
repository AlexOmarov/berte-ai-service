from business.util.ml_logger.logger import get_logger

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

logger = get_logger("APP")
llm = ChatOllama(model="qwen2:0.5b", num_gpu=1)

messages = [
    HumanMessage(
        content="What color is the sky at different times of the day? Respond using JSON"
    )
]

res = llm.invoke(messages)

logger.info(res)
