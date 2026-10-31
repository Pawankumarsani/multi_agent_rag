import os
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from typing import Literal
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import Tool
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
from langchain_sarvam import ChatSarvam

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["SARVAM_API_KEY"] = os.getenv("SARVAM_API_KEY")

rate_limiter = InMemoryRateLimiter(requests_per_second=0.2)
llm = ChatSarvam(
    model="sarvam-105b",
    rate_limiter=rate_limiter,
    max_retries=2,
)

def text_of(msg: BaseMessage) -> str:
    """Handle both str and list-of-blocks content."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


### generic function to create a retrieval tool
def make_retriever_tool_from_text(file, name, desc):
    docs=TextLoader(file,encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs =FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    
    def tool_func(query:str)->str:
        print(f"Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)
    return Tool(name=name,description=desc,func=tool_func)

tavily_tool=TavilySearchResults(max_results=5)

internal_tool_1=make_retriever_tool_from_text("internal_docs.txt",
                                              "InternalResearchNotes",
                                              "Search internal research notes for experimental results")

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )
    
### Reaearch agent and node

research_agent=create_agent(
    llm,
    tools=[internal_tool_1,tavily_tool],
    system_prompt=make_system_prompt("You can only do research. Use the tool that you are binded with, you can use both of them" \
    " You are working with a content writer colleague.")
)

## Reaearch node
def research_node(state:MessagesState)-> Command[Literal["blog_generator", "__end__"]]:
    result = research_agent.invoke(state)
    last_msg = result["messages"][-1]
    goto = get_next_node(last_msg, "blog_generator")

    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    shared_message = HumanMessage(
        content=last_msg.content, name="researcher"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": [shared_message],
        },
        goto=goto,
    )

### Blog write agent
blog_agent=create_agent(
    llm,
    tools=[],
    system_prompt=make_system_prompt(
        "You can only write a detailed blog. You are working with a researcher colleague."
    )
)

def blog_node(state: MessagesState) -> Command[Literal["researcher", "__end__"]]:
    result = blog_agent.invoke(state)
    last_msg = result["messages"][-1]
    goto = get_next_node(last_msg, "researcher")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    shared_message = HumanMessage(
        content=last_msg.content, name="blog_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": [shared_message],
        },
        goto=goto,
    )
    
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("blog_generator", blog_node)

workflow.add_edge(START, "researcher")
graph = workflow.compile()

if __name__ == "__main__":
    response = graph.invoke(
        {"messages": [HumanMessage(content="Write a detailed blog on transformer variants in production deployments")]},
        config={"recursion_limit": 12},
    )
    final = text_of(response["messages"][-1]).replace("FINAL ANSWER", "", 1).strip()
    print("\n--- FINAL BLOG ---\n")
    print(final)