# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o", temperature=0) 

import os
import pathlib
from dotenv import load_dotenv

# 指定 .env 文件路径
env_path = r'C:\GitRepo\langchain-academy\module-1\.env'


# 加载 .env 文件
load_dotenv(dotenv_path=env_path)
print(f"The AZURE_OPENAI_ENDPOINT is: {os.getenv('AZURE_OPENAI_ENDPOINT')}")

os.environ["LANGSMITH_PROJECT"] = "langchain-academy-agentic-rag"

from langchain_openai import AzureChatOpenAI

azure_openai_llm = AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True,
)

from langchain_openai import AzureOpenAIEmbeddings

azure_openai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
)

# embeddings.embed_query("Hello, world!")


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 本地向量存储的路径
# persist_directory = r"C:\GitRepo\langchain-academy\module-4\agentic-rag\db\chroma_db_azure_docs"
persist_directory = "../agentic-rag/db/chroma_db_azure_docs"

vectorstore_force_update = False

# 检查向量存储是否已存在
if os.path.exists(persist_directory) and os.path.isdir(persist_directory) and len(os.listdir(persist_directory)) > 0 and not vectorstore_force_update:
    print(f"正在从本地加载向量存储: {persist_directory}")
    # 直接从本地加载向量存储
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=azure_openai_embeddings,
        collection_name="rag-chroma-azure-docs-markdown"
    )
else:
    print("本地向量存储不存在，创建新的向量存储...")
    urls = [
        # "https://lilianweng.github.io/posts/2023-06-23-agent/",
        # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models",
        "https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter"

    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 创建新的向量存储并保存到本地
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        persist_directory=persist_directory,
        embedding=azure_openai_embeddings,
    )
    # 保存到本地
    # LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
    # vectorstore.persist()
    print(f"向量存储已保存到: {persist_directory}")

retriever = vectorstore.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
        "retrieve_azure_openai_docs",
        "Search and return information about Azure OpenAI models and content filtering.",
)

tools = [retriever_tool]

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    score: str

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = azure_openai_llm
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    print(response)
    # We return a list, because this will get added to the existing list
    # agent_response = AIMessage(content=response.content, name="agent")
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = azure_openai_llm
    response = model.invoke(msg)
    rewrite_response = AIMessage(content=response.content, name="rewrite")
    return {"messages": [rewrite_response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    # question = messages[0].content
    question = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    llm = azure_openai_llm

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    generate_response = AIMessage(content=response, name="generate")
    return {"messages": [generate_response]}


def grade_documents_node(state):
    """
    判断检索到的文档是否与问题相关，直接返回评分结果
    
    Args:
        state (dict): 当前状态
        
    Returns:
        dict: 包含评分结果的字典
    """
    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    model = azure_openai_llm
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))
    
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        
    return {"score": score}


def router(state):
    """
    根据grade_documents_node的评分结果决定路由
    
    Args:
        state (dict): 当前状态
        
    Returns:
        str: 下一个节点的名称
    """
    if state["score"] == "yes":
        return "generate"
    else:
        return "rewrite"


print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("grade_documents", grade_documents_node)  # 评估文档相关性的节点
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant

# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Retrieve调用完成后直接进入grade_documents节点
workflow.add_edge("retrieve", "grade_documents")

# 根据grade_documents的评分结果决定下一步
workflow.add_conditional_edges(
    "grade_documents",
    router,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()