"""
Autonomous AI Agent with Tool-Use Capabilities
================================================
An intelligent AI agent built with LangChain and GPT-4 that autonomously
selects and executes tools to complete multi-step reasoning tasks.
Uses FAISS for persistent memory and context management.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from tools import get_all_tools
from memory import FAISSMemoryManager

load_dotenv()


def create_agent():
    """Create and configure the autonomous AI agent."""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        max_tokens=2048,
    )

    tools = get_all_tools()

    system_prompt = """You are an autonomous AI assistant with access to multiple tools.
    You can search the web, perform calculations, analyze data, fetch weather information,
    query Wikipedia, manage files, translate text, and summarize content.
    
    Guidelines:
    - Break down complex tasks into smaller steps
    - Select the most appropriate tool for each step
    - Provide clear, structured responses
    - Use your memory to maintain context across conversations
    - If a tool fails, try an alternative approach
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )

    return agent_executor


def main():
    """Main entry point for the AI agent."""
    print("=" * 60)
    print("  Autonomous AI Agent with Tool-Use Capabilities")
    print("  Type 'quit' to exit | 'history' to view memory")
    print("=" * 60)

    agent = create_agent()
    memory_manager = FAISSMemoryManager()

    while True:
        user_input = input("\n🧑 You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("\n👋 Goodbye!")
            break
        if user_input.lower() == "history":
            memory_manager.show_history()
            continue

        try:
            response = agent.invoke({"input": user_input})
            result = response["output"]

            # Store in persistent FAISS memory
            memory_manager.store_interaction(user_input, result)

            print(f"\n🤖 Agent: {result}")

        except Exception as e:
            print(f"\n⚠️  Error: {str(e)}")
            print("Trying alternative approach...")


if __name__ == "__main__":
    main()
