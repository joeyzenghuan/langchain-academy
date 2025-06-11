import os
from typing import Dict, List, TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
env_path = r'C:\GitRepo\langchain-academy\module-1\.env'
load_dotenv(dotenv_path=env_path)

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define state
from langgraph.graph import MessagesState

class DebateState(MessagesState):
    current_speaker: str
    round: int
    topic: str

def format_message_history(messages: Sequence[AnyMessage]) -> str:
    """Format message history with speaker information"""
    history = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            if hasattr(msg, 'name'):
                history.append(f"{msg.name}: {msg.content}")
            else:
                history.append(f"Moderator: {msg.content}")
    return "\n".join(history)

# Define nodes
def get_biden_response(state: DebateState) -> Dict:
    """Get Biden's response"""
    print("\n=== Biden's Turn ===")
    print(f"Current Round: {state['round']}")
    print(f"Current Speaker: {state['current_speaker']}")
    print(f"Topic: {state['topic']}")
    
    messages = state["messages"]
    current_speaker = state["current_speaker"]
    round_num = state["round"]
    topic = state["topic"]
    
    system_message = f"""You are President Joe Biden in a debate with Donald Trump. 
    The debate topic is: {topic}
    This is round {round_num} of the debate.
    You are known for your calm demeanor, experience in government, and focus on unity.
    Respond in a way that reflects Biden's speaking style and policy positions.
    Keep your response concise and focused on the topic.
    Reply in Chinese."""
    
    # Create messages list with system message and conversation history
    messages = [
        SystemMessage(content=system_message),
        *messages
    ]
    
    print("\nBiden's Messages Input:")
    for m in messages:
        m.pretty_print()
    
    response = llm.invoke(messages)
    
    print("\n================ Biden's Response: ==================")
    print(response.content)
    
    # Convert AI response to HumanMessage with name
    human_response = HumanMessage(content=response.content, name="Biden")
    
    return {
        "messages": [human_response],
        "current_speaker": "Trump",
        "round": round_num
    }

def get_trump_response(state: DebateState) -> Dict:
    """Get Trump's response"""
    print("\n=== Trump's Turn ===")
    print(f"Current Round: {state['round']}")
    print(f"Current Speaker: {state['current_speaker']}")
    print(f"Topic: {state['topic']}")
    
    messages = state["messages"]
    current_speaker = state["current_speaker"]
    round_num = state["round"]
    topic = state["topic"]
    
    system_message = f"""You are Donald Trump in a debate with Joe Biden.
    The debate topic is: {topic}
    This is round {round_num} of the debate.
    You are known for your direct speaking style, focus on America First policies, and tendency to use nicknames.
    Respond in a way that reflects Trump's speaking style and policy positions.
    Keep your response concise and focused on the topic.
    Reply in Chinese."""
    
    # Create messages list with system message and conversation history
    messages = [
        SystemMessage(content=system_message),
        *messages
    ]
    
    print("\nTrump's Messages Input:")
    for m in messages:
        m.pretty_print()
    
    response = llm.invoke(messages)
    
    print("\n================ Trump's Response: ==================")
    print(response.content)
    
    # Convert AI response to HumanMessage with name
    human_response = HumanMessage(content=response.content, name="Trump")
    
    return {
        "messages": [human_response],
        "current_speaker": "Biden",
        "round": round_num + 1
    }

def should_continue(state: DebateState) -> str:
    """Determine whether to continue the debate"""
    print("\n=== Checking Debate Status ===")
    print(f"Current Round: {state['round']}")
    print(f"Current Speaker: {state['current_speaker']}")
    
    if state["round"] > 2:
        print("Debate ended: Maximum rounds reached")
        return END
    
    next_speaker = "trump_response" if state["current_speaker"] == "Trump" else "biden_response"
    print(f"Next Speaker: {next_speaker}")
    return next_speaker

# Create graph
workflow = StateGraph(DebateState)

# Add nodes
workflow.add_node("biden_response", get_biden_response)
workflow.add_node("trump_response", get_trump_response)

# Add edges
workflow.add_edge(START, "biden_response")  # Add entrypoint

# Add conditional edges
workflow.add_conditional_edges(
    "biden_response",
    should_continue,
    {
        "trump_response": "trump_response",
        END: END
    }
)

workflow.add_conditional_edges(
    "trump_response",
    should_continue,
    {
        "biden_response": "biden_response",
        END: END
    }
)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

def run_debate(topic: str) -> None:
    """Run a debate between Biden and Trump on the given topic"""
    print("\n=== Starting Debate ===")
    print(f"Topic: {topic}")
    
    config = {"configurable": {"thread_id": "debate"}}
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=f"Let's debate about {topic}", name="Moderator")],
        "current_speaker": "Biden",
        "round": 1,
        "topic": topic
    }
    
    print("\nInitial State:")
    print(f"Current Speaker: {initial_state['current_speaker']}")
    print(f"Round: {initial_state['round']}")
    print(f"Initial Message: {initial_state['messages'][0].content}")
    
    # Run debate
    print("\n=== Running Debate ===")
    output = graph.invoke(initial_state, config)
    
    # Print debate transcript
    print("\n************************* Final Debate Transcript *************************")
    for message in output["messages"]:
        if isinstance(message, HumanMessage):
            if hasattr(message, 'name'):
                print(f"\n===={message.name}====:\n {message.content}")
            else:
                print(f"\n====Moderator====:\n {message.content}")

if __name__ == "__main__":
    topic = input("Enter debate topic: ")
    run_debate(topic)
