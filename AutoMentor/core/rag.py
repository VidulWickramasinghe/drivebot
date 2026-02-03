from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import config

def load_rag_chain():
    """Initializes and returns the RAG chain."""
    
    # 1. Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL)
    
    # 2. Load the FAISS index
    if not config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {config.FAISS_INDEX_PATH}. Please run the ingestion script first.")
    
    vector_store = FAISS.load_local(str(config.FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks
    
    # 3. Initialize the LLM
    llm = Ollama(model=config.OLLAMA_MODEL)
    
    # 4. Setup conversational memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # 5. Define a custom prompt template
    custom_prompt_template = """
    You are AutoMentor, a knowledgeable and precise automotive AI assistant. 
    Use the following pieces of context to answer the user's question. 
    If you don't know the answer from the context provided, state that you don't have enough information, don't try to make up an answer.
    Always provide step-by-step instructions in a clear, numbered list if the query involves a procedure.
    Cite the source document if its metadata is available in the context.

    Context: {context}
    Question: {question}
    Answer:
    """
    
    QA_PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    # 6. Create the ConversationalRetrievalChain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=False
    )
    
    return rag_chain

if __name__ == '__main__':
    # For testing the RAG module directly
    # Ensure you have run ingestion first.
    print("Testing RAG chain...")
    chain = load_rag_chain()
    print("RAG chain loaded successfully.")
    
    # Example query
    query = "What is this document about?"
    result = chain.invoke({"question": query})
    
    print("\n--- Test Query ---")
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}")
    
    # Example follow-up query
    query2 = "Can you elaborate on that?"
    result2 = chain.invoke({"question": query2})
    
    print("\n--- Follow-up Query ---")
    print(f"Question: {query2}")
    print(f"Answer: {result2['answer']}")
