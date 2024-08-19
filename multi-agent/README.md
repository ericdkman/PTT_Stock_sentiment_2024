# Sentiment Analysis-langchain
Description

---
學習langchain-ai/langgraph的方式
實作multi_agent-agent_supervisor
---
```
https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb
```
### 概念
![image]

kkkkkkkkkkk

### 程式碼
```
pip install langchain-core langchain-openai langchain-community langchain-experimental langchain
```
LangChain 是一個強大的框架，專門用來簡化和擴展自然語言處理（NLP）工作流程。它的主要目標是透過將不同的語言模型、工具和資源整合到一個統一的系統中，幫助開發者輕鬆構建和管理複雜的 NLP 應用。
此外lanchain可與langsmith連接，並透過langsmith追蹤程式的運行流程
(ex. 每個階段input、output)

### langchain_core
處理對話狀態(HumanMessage)、管理記憶和上下文、解析輸出
### langchain_openai
提供LangChain 與 OpenAI API 對接操作的功能
### langchain_community
開源社群提供的擴展工具
### langchain
主要核心庫，建立對話系統、自然語言處理工作等應用
```
import getpass
import os
...
...
...
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from typing import Sequence, TypedDict, Annotated

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

# 設置環境變量
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("OPENAI_ORGANIZATION")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Optional, add tracing in LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"
```
### 定義代理(agent) 
創建代理節點。它將大型語言模型（LLM）與各種 API、數據處理模組結合，並使用一個系統提示（system_prompt）來生成代理人執行器。

```
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

# 定義代理
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor
```



