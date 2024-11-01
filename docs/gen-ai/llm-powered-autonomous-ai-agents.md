# Autonomous AI Agents

Autonomous AI Agents are goal-driven, self-executing software to plan, execute and priortize tasks to achieve a certain goal, where LLMs are used as **reasoning engines**.. They translate natural language prompts into actions and execute them.

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

### Vector Datatabases

#### Why Vector Databases needed in context of LLMs.

By default, the LLMs bases its responses on textual content it has ingested during training. The training data may not include company's private data resulting inaccurate answer. To make LLM base its answer on private unseen data, we can agument user query with more context. For example, a legal document, a technical manual, or search results. We can then instruct the model to generate its answer based on that data. 

To faster agument data relevant to user query, you can use 'Search Engines'. But they are only capable of keyword search.

Vector databases provide efficient and quicker way to augument models with relevant context informaton. They have advanced indexing and search algorithms top perform efficient similiarity searches. 

#### What is a vector?
A vector is an array of numbers like [0, 1, 2, 3, 4, … ]. Vector can represent more complex objects such as words, sentences, images, and audio files in an embedding.

#### What is Word Embedding? 
In the context of large language models, embeddings represent text as a dense vector of numbers to capture the meaning of words. They map the semantic meaning of words together or similar features into vectors. These embeddings can then be used for search engines, recommendation systems, and generative AIs such as ChatGPT. 

More concepts,
- [What is Tokenization?](https://ankur3107.github.io/blogs/intro-to-tokenization-using-openai-chatgpt/)
- [Which tokenization method ChatGPT uses?](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
- [What is difference between Tokens & Word Embeddings?](https://medium.com/@saschametzger/what-are-tokens-vectors-and-embeddings-how-do-you-create-them-e2a3e698e037)

- Vector databases can measure the distance between two vectors, which defines their relationship. Small distances suggest high relatedness, while larger distances suggest low relatedness.
- Vector databases enhance the memory of LLMs thorugh context injection. Prompt augmentation feeds LLMs with contextual data.

<img src="assets/vector-db-llms.png" width="50%" height="50%" alt="Vector DBs"/>

Most popular vector databases are,

| Pinecone | Chroma | Weaviate |
| --- | --- | --- |
| Pinecone is a cloud-based vector database designed to efficiently store, index and search extensive collections of high-dimensional vectors. | Chroma is an open source vector database that provides a fast and scalable way to store and retrieve embeddings. <br/>Chroma is designed to be lightweight and easy to use, with a simple API and support for multiple backends, including RocksDB and Faiss (Facebook AI Similarity Search) | Weaviate is an open source vector database designed to build and deploy AI-powered applications. Weaviate’s key features include support for semantic search and knowledge graphs and the ability to automatically extract entities and relationships from text data.|

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
