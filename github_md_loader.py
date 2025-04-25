from langchain_community.document_loaders import WebBaseLoader
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# 加载 .env 文件
env_path = r'C:\GitRepo\langchain-academy\module-1\.env'
load_dotenv(dotenv_path=env_path)

def load_markdown_from_github(github_url):
    """
    Load markdown content from a GitHub URL using LangChain's WebBaseLoader
    
    Args:
        github_url (str): URL to the GitHub markdown file
        
    Returns:
        list: List of Document objects containing the markdown content
    """
    # Create a WebBaseLoader with the GitHub URL
    loader = WebBaseLoader(github_url)
    
    # Load the content from the URL
    documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s) from {github_url}")
    
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks for processing
    
    Args:
        documents (list): List of Document objects
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of split Document objects
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Split into {len(split_docs)} chunks")
    
    return split_docs

def create_vector_store(documents):
    """
    Create a vector store from documents
    
    Args:
        documents (list): List of Document objects
        
    Returns:
        Chroma: Vector store object
    """
    from langchain_openai import AzureOpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # Initialize Azure OpenAI embeddings
    azure_openai_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    )
    
    # Create a vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=azure_openai_embeddings,
    )
    
    return vectorstore

def query_document(vectorstore, query_text):
    """
    Query the vector store
    
    Args:
        vectorstore: Vector store object
        query_text (str): Query text
        
    Returns:
        list: List of relevant documents
    """
    # Create a retriever
    retriever = vectorstore.as_retriever()
    
    # Query the retriever
    docs = retriever.invoke(query_text)
    
    print(f"Found {len(docs)} relevant document(s)")
    
    return docs

if __name__ == "__main__":
    # URL from the user query
    github_url = "https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-services/openai/concepts/models.md"
    
    # Convert to raw content URL for direct access
    # This transforms GitHub URLs to raw content URLs for direct access
    raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    # Load documents from the URL
    documents = load_markdown_from_github(raw_url)
    
    # Print metadata and preview of the content
    if documents:
        print("\nDocument Metadata:")
        print(documents[0].metadata)
        
        print("\nContent Preview (first 500 chars):")
        print(documents[0].page_content[:500])
        
        # Split documents
        split_docs = split_documents(documents)
        
        # Create vector store
        vectorstore = create_vector_store(split_docs)
        
        # Example query
        query = "What are the newest Azure OpenAI models available?"
        relevant_docs = query_document(vectorstore, query)
        
        if relevant_docs:
            print("\nRelevant Document Content:")
            for i, doc in enumerate(relevant_docs):
                print(f"\n--- Document {i+1} ---")
                print(doc.page_content[:300] + "...") 