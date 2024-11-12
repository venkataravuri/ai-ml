
## LangChain, LangGraph, LangFuse

https://blog.langchain.dev/semi-structured-multi-modal-rag/

https://www.together.ai/blog/multimodal-document-rag-with-llama-3-2-vision-and-colqwen2

LangChain introduces **modular abstractions** for essential components to interact with language models. These components include:

- **Schema**: Schema pertains to the fundamental data types and structures utilized across the codebase. Text, ChatMessages, Examples, and Document.
- **Models**: Encompasses three distinct model types: Language Models, Chat Models, and Text Embedding Models.
- **Prompt** Templates: A structured framework with placeholders, which can be filled in with specific details or examples
- **Indexes**: Incorporate supplementary context within the prompt. 
- **Memory**: Storage and retrieval of chat history.
- **Chains**: Linking multiple LangChain components
- **Agents**: Agent Executor, Tools



#### Tools

Tools are interfaces that an agent can use to interact with the world. These tools can be generic utilities (e.g. search), other chains, or even other agents.

Tools can be,
- [Shell Tool](https://python.langchain.com/docs/integrations/tools/bash)
- [Search Tools](https://python.langchain.com/docs/integrations/tools/ddg)
- [Requests](https://python.langchain.com/docs/integrations/tools/requests)
- Others ...

https://colab.research.google.com/github/langfuse/langfuse-docs/blob/main/cookbook/integration_langgraph.ipynb
