
# Trying out LangChain with agents and tools
# Some tools I can think of
# - Tool to control home automation - turn on or off fan, control speed
# - Tool to give spelling test for didi
# = Tool to search web
# - agent that can understand if and use the correct tool







# Chat with your documents using LLM
#
# I tried both LM Studio and Ollama as local backend LLMs to LangChain
# Pros and cons of using local LLMs
# Pros: 
# - Your data doesn't leak out to the internet
# - You save some money by not subscribing to OpenAI chatgpt API service
# Cons:
# - Slower inference time

# I prefer LM Studio since has a wonderful interface for me to chat, and 
# I can also run it as a API backend and it supports the OpenAI API protocol

# Different LLMs models may behave differently to the prompt template
# So you may have to adjust the prompt as necessary
# Refer to documentation related with the model for more details

# Useful references
# - https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
# - https://medium.com/ai-in-plain-english/using-langchain-chains-and-agents-for-llm-application-development-d538f6c70bc6

from langchain import hub
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, BSHTMLLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType, tool, AgentExecutor, create_tool_calling_agent
from datetime import date
import streamlit as st
import os

DOC_FOLDER = r".\docs"  # documents to be vectorized and indexed
DB_FOLDER = r".\.db"    # chroma db
FILE_EXT = (".md", ".html", ".pdf")  # file extensions supported
# configuration of the LLM
# todo: import this from a json file
RAG_CONFIGS = [
    {   "name":"remote",
        "base_url":"http://192.168.1.66:1234/v1",   # this is the endpoint of a MacBook Pro that runs LM Studio remotely
        "model":"ollama",
        "api_key":"123"
    },
    {   "name":"local_openai",
        "base_url":"http://127.0.0.1:1234/v1",  # this is the endpoint of my Intel based Window Notbook that runs LM Studio locally
        "model":"ollama",
        "api_key":"123"
    },
    {
        "name":"local_ollama",
        "base_url":"http://127.0.0.1:11434",    # this is the endpoint of my Intel based Window Notbook that runs Ollama locally
        "model":"mistral",
    }
]

MODE = "local_ollama"  #select the API endpoint of the local LLM
CONFIG = [c for c in RAG_CONFIGS if c["name"] == MODE][0]



# upserts docs to the chroma db that referenced the documents
# it skips files that have already been added by using the file path as unqiue identifier (id) in chroma
# returns the unique ids and splitted docs 
def split_docs(vectorstore, path=DOC_FOLDER, file_ext=FILE_EXT):
    
    # get list of unique ids of docs from chroma db
    ids = vectorstore.get()["ids"]
    # Add the documents to the vectorstore after splitting them into chunks
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len
    )

    # Iterate through all subdirectories
    ret_docs = []
    ret_ids = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(file_ext):
                filename = os.path.join(root,file)
                # print(filename)
                # only include files that are not found in chroma yet
                found = [f for f in ids if filename in f]
                print("found", found)
                # found = find_ids(filename, ids)
                if found == []:
                    # print(f"cannot find {filename}")
                    try:
                        # loader = BSHTMLLoader(filename)     # use beautiful soup loader if using .html
                        # if os.path.exists(filename): 
                        #     print(f"found {filename}")
                        if filename.endswith(".pdf"):
                            loader = PyPDFLoader(filename)
                        elif filename.endswith(".html"):
                            loader = BSHTMLLoader(filename)
                        else:
                            loader = TextLoader(filename, autodetect_encoding=True)

                        doc = loader.load()
                    except UnicodeDecodeError as e:
                        print(f"Error decoding the file: {e}")
                        continue
                    except Exception as e:
                        print(f"Error loading the file: {filename} - {e}")
                        continue
                        
                    i = 0
                    chunks = text_splitter.split_documents(doc)
                    print("number of chunks: ",chunks.__len__())
                    print(chunks)
                    for chunk in chunks:
                        i += 1
                        ret_docs.append(chunk)
                        chunk_id = f"{filename} ({str(i)})"
                        ret_ids.append(chunk_id)
                        print(f"added {chunk_id}")
                else:
                    print(f"skipped, found {filename}")
    return ret_docs, ret_ids


def main():

    @tool
    def time(text: str) -> str:
        """Returns todays date, use this for any \
        questions related to knowing todays date. \
        The input should always be an empty string, \
        and this function will always return todays \
        date - any date mathematics should occur \
        outside this function."""
        return str(date.today())


    @tool
    def ask_docs(question: str) -> str:
        """Ask questions about the documents."""

        template = """You are a helpful and chatty personal assistant. Answer the following question using the context politely. 
        In your answer, do not say it is based on the context and do not include any additional notes.
        ====CONTEXT====
        Context: {context}
        ====QUESTION====
        User: {question}
        =====ANSWER====
        Assistant: """

        prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
                )
        qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    # chain_type="refine",
                    chain_type='stuff',
                    # chain_type="map_reduce",                    
                    # chain_type="map_rerank",                    
                    retriever=st.session_state["retriever"],
                    verbose=True,
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": prompt,}
        )
        answer = qa_chain(question)["result"]
        return answer

    tools = [time, ask_docs]

    st.title("Chatbot")

    # parser = PydanticOutputParser(pydantic_object=Response)
    
    loaded = st.session_state.get("loaded", False)
    if not loaded:

        # Define custom CSS for word wrapping
        custom_css = """
        <style>
            .streamlit-text-container {
                white-space: pre-wrap;
                font-size: 14px;
            }
        </style>
        """

        # Inject the custom CSS
        st.markdown(custom_css, unsafe_allow_html=True)

        ## Add document along with unique IDs so we don't include the same documents again
        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_FOLDER,
                            embedding_function=embedding)             

        docs = []
        ids = []
        docs, ids = split_docs(vectorstore)
        if not ids == []:
            vectorstore = vectorstore.from_documents(documents=docs, embedding=embedding, ids=ids, persist_directory=DB_FOLDER)
            vectorstore.persist()

        retriever = vectorstore.as_retriever()
        st.session_state["retriever"] = retriever
        # test_retrival = vectorstore.similarity_search_with_score("Who are my family?", 3)
        # print(f"test vector retrival: {test_retrival}")

        llm = ChatOllama(model="llama3", temperature=0)


        # if MODE == "local_ollama":
        #     llm = Ollama(
        #         base_url=CONFIG["base_url"],
        #         model=CONFIG["model"],
        #         temperature=0,
        #         verbose=True,
        #         callback_manager=CallbackManager(
        #             [StreamingStdOutCallbackHandler()]),
        #         )
        # else:
        #     llm = OpenAI(
        #         base_url=CONFIG["base_url"],
        #         model=CONFIG["model"],
        #         openai_api_key=CONFIG["api_key"],
        #         temperature=0,
        #         verbose=True,
        #         callback_manager=CallbackManager(
        #             [StreamingStdOutCallbackHandler()]),
        #         )


        # Get the prompt to use - you can modify this!
        prompt = hub.pull("hwchase17/openai-tools-agent")
        prompt.pretty_print()

        # Construct the tool calling agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



        # # Initialize the agent
        # agent= initialize_agent(
        #     [time], 
        #     llm, 
        #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        #     handle_parsing_errors=True,
        #     verbose = True)

        # Call the agent
        response = agent_executor.invoke(
            {
                "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
            }
        )
        print(response["output"])

        # st.session_state["qa_chain"] = qa_chain
        st.session_state["agent"] = agent_executor
        st.session_state["loaded"] = True

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_input := st.chat_input("You:", key="user_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state["agent"].invoke(
                    {"input": user_input}) 
                print(response)
                st.write(response['result']) 
        message = {"role": "assistant", "content": response['result']}
        st.session_state.messages.append(message)            

if __name__ == "__main__":
    main()

