# AI Agents

Language model powered AI agents and state machines have emerged as a promising design pattern for creating flexible and effective ai-powered products. Agents use LLMs as general-purpose problem-solvers, connecting them with external resources to answer questions or accomplish tasks. 

The structure and design pattern of an LLM program is commonly called **cognitive architecture**.

## Agent Architecture

The key components in AI Agentic framework are:

<img src="agent-framework.png" height="70%" width="70%" />

- **Planning**
    - _Subgoal and decomposition_: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
    - _Reflection and refinement_: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.

- **Memory**
    - _Short-term memory_: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model to learn.
    - _Long-term memory_: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.

- **Tool use**
    - The agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including current information, code execution capability, access to proprietary information sources and more.
---

### ReAct Agent

The ReAct agent is a great prototypical design for this, as it prompts the language model using a repeated thought, act, observation loop:

    Thought: I should call Search() to see the current score of the game.
    Act: Search("What is the current score of game X?")
    Observation: The current score is 24-21
    ... (repeat N times)

This takes advantage of _Chain-of-thought prompting to make a single action choice per step_.

### Plan-And-Execute

**Plan-and-Solve** Prompting is a simple planning agent architecture consists of two basic components:

- A **planner**, which prompts an LLM to generate a multi-step plan to complete a large task.
- **Executor(s)**, which accept the user query and a step in the plan and invoke 1 or more tools to complete that task.

<img src="plan-execute.png" height="60%" width="60%" />

Once execution is completed, the agent is called again with a re-planning prompt, letting it decide whether to finish with a response or whether to generate a follow-up plan.

#### References

- https://blog.langchain.dev/planning-agents/
