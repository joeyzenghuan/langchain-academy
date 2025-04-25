
import os
from dotenv import load_dotenv

# 指定 .env 文件路径
# env_path = r'C:\GitRepo\langchain-academy\module-1\.env'
env_path = '../studio/.env'

# env_path = r'C:\GitRepo\langchain-academy\module-4\studio\.env'
# 加载 .env 文件
load_dotenv(dotenv_path=env_path)
print(f"The AZURE_OPENAI_ENDPOINT is: {os.getenv('AZURE_OPENAI_ENDPOINT')}")



os.environ["LANGSMITH_PROJECT"] = "langchain-academy-agentic-rag"

# from langchain_openai import AzureChatOpenAI

# azure_openai_llm = AzureChatOpenAI(
#     azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
#     api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     streaming=True,
# )

from langchain_openai import AzureOpenAIEmbeddings

azure_openai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)



from langchain_community.document_loaders import DirectoryLoader,TextLoader
loader = DirectoryLoader("data", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
docs


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs)
doc_splits
# # Add to vectorDB
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma-azdocs-test",
#     # embedding=OpenAIEmbeddings(),
#     embedding=azure_openai_embeddings,
# )

persist_directory = "db/chroma_db_azure_docs"
# 创建新的向量存储并保存到本地
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma-azure-docs-markdown",
    persist_directory=persist_directory,
    embedding=azure_openai_embeddings,
)
# 保存到本地
# LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
# vectorstore.persist()
print(f"向量存储已保存到: {persist_directory}")
