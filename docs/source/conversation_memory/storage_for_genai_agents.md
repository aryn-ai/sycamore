# Storage for Generative AI Agents

Developers use generative AI models to do all sorts of tasks, and can combine these models through a technique called Chain-of-Thought (CoT) reasoning. Instead of asking an LLM to solve a problem, you tell it what tools it has at its disposal and ask it to walk through the steps it would take to solve the problem. This LLM can communicate with other generative AI models, called Agents, to carry out these specific tasks. In this way, you can break down complicated tasks into specific tasks, with each task type carried out by a specific agent.


![Untitled](imgs/ConversationMemoryMultiAgent.jpg)


With multi-agent applications that are driven by natural language requests, it is essential to have a single source of truth for the conversation history. Multiple agents should be able to read the history of the same conversation, know where each interaction came from, and add their interactions in the CoT pipeline. Not only does this allow an agent to use the context of previous interactions, but also to reference the logic and authority of other agents involved in the conversation.

Sycamore's conversation memory APIs can be used to store this data for generative AI agents. For an example, visit [this tutorial](../tutorials/using_aryn_with_langchain.md).
