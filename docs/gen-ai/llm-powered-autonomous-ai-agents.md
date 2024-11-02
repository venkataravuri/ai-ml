# Autonomous AI Agents

Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs necessary to perform the action. After executing actions, the results can be fed back into the LLM to determine whether more actions are needed, or whether it is okay to finish.

The are goal-driven, self-executing software to plan, execute and priortize tasks to achieve a certain goal.

Here are few browser-based autonomous LLM agents,
- [AgentGPT](https://agentgpt.reworkd.ai/) - Assemble, configure, and deploy autonomous AI Agents in your browser.
- [BabyAGI](https://babyagi.org/) - AI-powered task management system that uses OpenAI and Pinecone APIs to create, prioritize, and execute tasks.
- [Godmode.space](https://godmode.space/) - Explore the Power of Generative Agents on Godmode.space. Inspired by Auto-GPT and BabyAGI. Supports GPT-3.5 & GPT-4.
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)(non-browser based) - A program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, Auto-GPT pushes the boundaries of what is possible with AI.

Autonomous agents achieve specified goals by breaking it into tasks, execute them independently without human intervention. Agents use a LLM to determine which actions to take and in what order. The agent creates a **Chain-of-Thought** sequence on the fly by decomposing the user request.


**Table of Contents**
- [Introduction](#introduction)
- [What are Autonomous LLM Agents?](#what-are-autonomous-llm-agents)
- [How do you Autonomous LLM Agents work?](#how-do-autonomous-llm-agents-work)
- [How to build Autonomous LLM Agents?](#how-do-autonomous-llm-agents-work)
    - Task Planning
    - Tools
    - Memory
- [Frameworks & Libraries](#frameworks--libraries)
    - [Vector Databases](#vector-datatabases)
- [Demo 1: Autonomous Travel Agent](#example-autonomous-travel-agent)
- [Demo 2: Tanzu Kubernetes Autonomous AI Agent](#example-tanzu-kubernetes-autonomous-ai-agent)
- [Self-Learning Tutorials](#self-learning-tutorials)

### Frameworks & Libraries

Popular frameworks to build LLM powered applications,

| <img src="https://python.langchain.com/img/parrot-chainlink-icon.png" width="12%" height="12%"/> LangChain | <img src="https://cdn-images-1.medium.com/v2/resize:fit:1200/1*_mrG8FG_LiD23x0-mEtUkw.jpeg" width="8%" height="8%"/> LlamaIndex | <img src="https://avatars.githubusercontent.com/u/128686189?s=48&v=4" width="8%" height="8%"/> Chainlit |
| --- | --- | --- |
|[LangChain](https://www.langchain.com/) is an open-source Python library that enables anyone who can write code to build LLM-powered applications. <br/>The package provides a generic interface to many foundation models, enables prompt management, and acts as a central interface to other components like prompt templates, other LLMs, external data, and other tools via agents.| [LlamaIndex](https://www.llamaindex.ai/) is an open-source project that provides a simple interface between LLMs and external data sources like APIs, PDFs, SQL etc.<br/>It provides indices over structured and unstructured data, helping to abstract away the differences across data sources. It can store context required for prompt engineering, deal with limitations when the context window is too big, and help make a trade-off between cost and performance during queries. | [Chainlit](https://docs.chainlit.io/overview) library is similar to Python’s Streamlit Library.<br/>This library is seamlessly integrated with LangFlow and LangChain(the library to build applications with Large Language Models), which we will do later in this guide.<br/>Chainlit even allows for visualizing multi-step reasoning.

[Langflow](https://langflow.org/) - A low-code solution to build LLM powered Apps.

LangChain framework components,

<img src="assets/langchain-components.png" width="60%" height="60%" alt="LangChain Components"/>

## Example: Autonomous Travel Agent

An autonomous travel agent powered by LLMs such as ChatGPT to automate airline ticket booking process.

It uses ReAct prompting techinque. It allow the model to “reason” (with a chain-of-thoughts) and “act” (by being able to use a tool from a predefined set of tools, such as being able to search the internet).

#### Demonstration

[Source Code](https://github.com/venkataravuri/ai-ml/tree/master/llm-powered-autonomous-agents)

#### Code Walkthrough

[ReAct Pattern implementation using LangChain](https://github.com/venkataravuri/ai-ml/blob/master/llm-powered-autonomous-agents/pages/1_%E2%9B%93%EF%B8%8F_React.py)

## Example: Tanzu Kubernetes Autonomous AI Agent

A sample autonoums agent that peforms Tanzu Kubernetes automation activities.

### Demo & Code Walkthrough

[Google Colab Notebook](https://colab.research.google.com/drive/11VbZ0T7HI-5I_tU2Nl1XH78LkJUN8mFg)

## Self-Learning Tutorials

- [LangChain tutorial #1: Build an LLM-powered app in 18 lines of code](https://blog.streamlit.io/langchain-tutorial-1-build-an-llm-powered-app-in-18-lines-of-code/)
- [LangChain tutorial #4: Build an Ask the Doc app](https://blog.streamlit.io/langchain-tutorial-4-build-an-ask-the-doc-app/)

## AI Roles

After LLMs are out in market, a new role "AI Engineer" is emerging, with accountability to productize large language models. AI Engineer role is to integrate developed models into software products.

<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa81555af-0b76-4a61-9b53-595e3d47580a_1005x317.png" width="75%" height="75%" alt="AI Engineer"/>

Source: [Latent Space](https://www.latent.space/p/ai-engineer)
