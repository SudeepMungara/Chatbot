import os
import openai
import chainlit as cl
from PyPDF2 import PdfReader
from chainlit.types import AskFileResponse
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
os.environ['OPENAI_API_KEY'] = "API KEY"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()
def process_file(file:AskFileResponse):
    import tempfile
    
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs
    
def get_docsearch(file: AskFileResponse):
    docs = process_file(file)
    
    cl.user_session.set("docs", docs)

    docsearch = FAISS.from_documents(
        docs, embeddings
    )
    return docsearch
    

@cl.on_chat_start
async def on_chat_start():
    files = None
    
    while files == None:
        files = await cl.AskFileMessage(
            content="To get started please upload text or PDF file and question the file",
            accept=["text/plain","application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
        
    file = files[0]
    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()
    
    docsearch = await cl.make_async(get_docsearch)(file)
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)
    
@cl.on_message
async def main(message:cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    txt_elements = []
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            txt_elements.append(cl.Text(content=source_doc.page_content, name=source_name))
            source_names = [text_el.name for text_el in txt_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
            
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = txt_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=txt_elements).send()
