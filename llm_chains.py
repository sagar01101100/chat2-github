from prompt_templates import memory_prompt_template
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
import chromadb
import yaml

# Load configuration parameters from the config.yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Function to create an LLM instance based on the provided model path, type, and configuration
def create_llm(model_path = config["model_path"]["large"], model_type = config["model_type"], model_config = config["model_config"]):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

# Function to create HuggingFace embeddings, used for vector-based operations
def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)

# Function to create a memory object that stores conversation history
def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

# Function to create a prompt template from a given template string
def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

# Function to create an LLM chain with the provided LLM, chat prompt, and memory
def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

# Loads a normal chat chain without PDF or vector database integration
def load_normal_chain(chat_history):
    return chatChain(chat_history)

# Function to load a persistent vector database using Chroma for storing embeddings
def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient("chroma_db")  # Points to a local or cloud-based Chroma database

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="pdfs",  # Name of the collection to retrieve vectors
        embedding_function=embeddings, # Embedding function to store and compare vectors
    )
    return langchain_chroma

# Loads a PDF chat chain with vector database integration
def load_pdf_chat_chain(chat_history):
    return pdfChatChain(chat_history)

# Creates a retrieval chain that combines the LLM, memory, and vector retriever
def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever=vector_db.as_retriever(kwargs={"k": 3}))  # Retrieve top 3 matches

# PDF Chat Chain Class
class pdfChatChain:

    def __init__(self, chat_history):
        # Initialize memory to store chat history
        self.memory = create_chat_memory(chat_history)
        # Load embeddings and create a vector database
        self.vector_db = load_vectordb(create_embeddings())
        
        # Create an LLM instance for answering questions
        llm = create_llm()
        #chat_prompt = create_prompt_from_template(memory_prompt_template)
        # Create a retrieval chain that combines the LLM, memory, and vector retriever
        self.llm_chain = load_retrieval_chain(llm, self.memory, self.vector_db)

    # Run method to process user input and fetch answers
    def run(self, user_input):
        print("Pdf chat chain is running...")
        return self.llm_chain.run(
            query = user_input,
            history=self.memory.chat_memory.messages, # Include past messages
            stop=["Human:"] # Define stopping criteria
        )

# Normal Chat Chain Class
class chatChain:

    def __init__(self, chat_history):
        
        # Initialize memory to store chat history
        self.memory = create_chat_memory(chat_history)
        
        # Create an LLM instance for generating responses
        llm = create_llm()
        # Create a chat prompt template to store chat history
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        # Combine LLM, memory, and prompt to form a chain
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input = user_input, history=self.memory.chat_memory.messages ,stop=["Human:"])