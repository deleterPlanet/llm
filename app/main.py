from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.llm import LLMChain
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = "sk-inZSm"
OPENAI_API_KEY += "4EIUAuuTheOACwiT3"
OPENAI_API_KEY += "BlbkFJLYrOPNc"
OPENAI_API_KEY += "UZsKB8kCIbZ1N"
MESSAGE_BUF_LIMIT = 2000
HELP_INFO_LIMIT = 3000
CONTEXT_LIMIT = MESSAGE_BUF_LIMIT + HELP_INFO_LIMIT
PORT = 8080

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings_model)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
redis_backed_dict = {}

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

app = FastAPI()


@app.post("/message")
def message(msg: str, user_id: str):
    docs = db.similarity_search(msg, 3)

    help_info_len = 0
    for doc in docs:
        help_info_len += len(doc.page_content)

    while help_info_len > HELP_INFO_LIMIT:
        help_info_len -= len(docs.pop().page_content)

    info = ""
    for doc in docs:
        info += doc.page_content

    memory = redis_backed_dict.get(
        user_id,
        ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=1)
    )
    prompt.messages[0] = SystemMessagePromptTemplate.from_template(
            "Pretend, that you are a call center operator at the 'Tinkoff bank'."
            "When user contacts you first, you greet them"
            "You can help with the answers to the questions"
            f"Here the information you should refer to first:{info}"
            "Now i am a user and you must greet me and answer my questions."
        )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    ai_message = conversation({"question": msg})
    redis_backed_dict.update({user_id: memory})
    return {"message": ai_message["text"]}


#if __name__ == "__main__":
#    uvicorn.run(app, host="localhost", port=PORT)
