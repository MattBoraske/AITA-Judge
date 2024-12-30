# AITA Evaluation Agent 🤖⚖️

An LLM agent that evaluates "Am I The A**hole" (AITA) conflicts with a nuanced understanding that is backed by real-life evidence.

## 🌟 Overview

This agent leverages [RAG (Retrieval-Augmented Generation)](https://docs.llamaindex.ai/en/stable/understanding/rag/) to analyze AITA scenarios by drawing insights from a database of nearly 40k conflicts sourced from the [r/amithea**hole](https://www.reddit.com/r/AmItheAsshole/) subreddit.

## 🛠️ Technical Architecture

Under the hood, the agent is implemented as a [llamaindex workflow](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) that progressively refines its judgment:

1. The agent is passed a new conflict.
2. Similar conflicts are retrieved from the AITA database.
- **Embeddings**: OpenAI's [`text-embedding-3-small`](https://platform.openai.com/docs/guides/embeddings/) model
- **Vector Storage**: [Pinecone](https://www.pinecone.io/) vector database
- **Reranking**: [Cohere's reranker](https://cohere.com/rerank) for enhanced relevance
1. The agent is given the top-ranked retrieved conflict and uses it as context when evaluating the input conflict.
2. The agent is subsequentally given the conflict and uses it to refine its existing answer. 
3. The process continues until all retrieved contexts are considered.

## 🔄 Workflow Visualization

```mermaid
graph TD
    A[New AITA Conflict] --> B[Generate Embeddings]
    B --> C[Query Vector DB]
    C --> D[Retrieve Similar Conflicts]
    D --> E[Rerank Conflicts]
    E --> F[Initial Eval]
    

    subgraph "Sequential Refinement"
        F --> G[Refine Using Conflict 1]
        G --> H[Refine Using Conflict N]
    end
    
    H --> J[Final Judgment]

    %% Style definitions
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,color:black;
    classDef refinement fill:#e6f3ff,stroke:#333,stroke-width:2px,color:black;
    class F,G,H,I,J refinement;
```
## 🚀 Deployment

Deployment using [llama-deploy](https://github.com/run-llama/llama_deploy) is a WIP. 