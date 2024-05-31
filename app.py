### Imports:
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

### Constants:
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0

SESSION_ID = "mysessionid"
LANGUAGE = "Spanish"

model = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

store = {} # chat history store

# Limits message history to k entries
def filter_messages(messages, k=10):
    return messages[-k:]

# Gets the message history for the given session id
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}")
    ]
)

chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["history"]))
    | prompt
    | model
)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": SESSION_ID}}

for r in with_message_history.stream(
    {
        "input": [HumanMessage(content="hi, I'm Todd")],
        "language": LANGUAGE
    }, config=config):
    print(r.content, end="|")


for r in with_message_history.stream(
    {
        "input": [HumanMessage(content="what's my name?")],
        "language": LANGUAGE,
    }, config=config):
    print(r.content, end="|")
