
"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from deep_research_from_scratch.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState

from trustcall import create_extractor

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %#d, %Y")

# ===== CONFIGURATION =====

# Initialize model
# model = init_chat_model(model="openai:gpt-4.1", temperature=0.0)
import os
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import  ChatNVIDIA
model=ChatGroq(groq_api_key=GROQ_API_KEY,model='openai/gpt-oss-120b',max_tokens=8000)
# model=ChatNVIDIA(model='qwen/qwen3-next-80b-a3b-thinking',NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY"))
from langchain.output_parsers import PydanticOutputParser



# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # Set up structured output model
    # structured_output_model = model.with_structured_output(ClarifyWithUser)
    # structured_output_model=model.bind_tools([ClarifyWithUser])
    # structured_output_model = create_extractor(
    #     model,
    #     tools=[ClarifyWithUser],
    #     tool_choice="ClarifyWithUser"
    # )





    # Invoke the model with clarification instructions
    response = model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ]
    )

    # print(response)

    parser = PydanticOutputParser(pydantic_object=ClarifyWithUser)
    response = parser.parse(response.content)


    # print(response)

    # print(response.need_clarification)

    # res=response['responses'][0]

    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    # structured_output_model = model.with_structured_output(ResearchQuestion)
    # structured_output_model = model.bind_tools([ResearchQuestion])


    # structured_output_model = create_extractor(
    #     model,
    #     tools=[ResearchQuestion],
    #     tool_choice="ResearchQuestion"
    # )

    # Generate research brief from conversation history
    response = model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ]
    )

    # print(response)

    parser = PydanticOutputParser(pydantic_object=ResearchQuestion)
    response = parser.parse(response.content)


    # print(response)


    # res=response['responses'][0]

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# ===== GRAPH CONSTRUCTION =====

# Build the scoping workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# Compile the workflow
scope_research = deep_researcher_builder.compile()
