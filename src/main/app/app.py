from business.util.ml_logger.logger import get_logger

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import os

logger = get_logger("APP")
llm = ChatOllama(model="llama3:8b", num_gpu=10000, base_url=os.environ['OLLAMA_HOST'])

messages = [
    HumanMessage(
        content="Describe a glass of wine which stands on a table outside the restaurant"
    )
]

res = llm.invoke(messages)

logger.info(res)
