# Vector Datatabases
LLMs base their responses on textual content it has ingested during training. The training data may not include company's private data resulting inaccurate answer. To make LLM base its answer on private unseen data, we can agument user query with more context e.g., a legal document, a technical manual, or search results. With this additional information we can instruct the model to generate its answer based on that data. 

To faster agument data relevant to user prompt, you can use 'Search Engines'. But they are only capable of keyword search.

Vector databases provide efficient and quicker way to augument models with relevant context informaton. They have advanced indexing and search algorithms top perform efficient similiarity searches. 

- Vector databases can measure the distance between two vectors, which defines their relationship. Small distances suggest high relatedness, while larger distances suggest low relatedness.
- Vector databases enhance the memory of LLMs thorugh context injection. Prompt augmentation feeds LLMs with contextual data.

#### What is a vector?
A vector is an array of numbers like [0, 1, 2, 3, 4, … ]. Vector can represent more complex objects such as words, sentences, images, and audio files in an embedding.

#### What is Word Embedding? 
In the context of large language models, embeddings represent text as a dense vector of numbers to capture the meaning of words. They map the semantic meaning of words together or similar features into vectors. These embeddings can then be used for search engines, recommendation systems, and generative AIs such as ChatGPT. 

Most popular vector databases are,

| Pinecone | Chroma | Weaviate |
| --- | --- | --- |
| Pinecone is a cloud-based vector database designed to efficiently store, index and search extensive collections of high-dimensional vectors. | Chroma is an open source vector database that provides a fast and scalable way to store and retrieve embeddings. <br/>Chroma is designed to be lightweight and easy to use, with a simple API and support for multiple backends, including RocksDB and Faiss (Facebook AI Similarity Search) | Weaviate is an open source vector database designed to build and deploy AI-powered applications. Weaviate’s key features include support for semantic search and knowledge graphs and the ability to automatically extract entities and relationships from text data.|

<img src="assets/vector-db-llms.png" width="50%" height="50%" alt="Vector DBs"/>

## Vector Similarity Search Algorithms

Vector similarity search looks for vectors that are closest in terms of distance (e.g., Euclidean distance or Cosine similarity) to the query vector. The distance between two vectors defines their relationship,
- Small distances suggest high relatedness.
- While, Larger distances suggest low relatedness.

<img src="https://dz2cdn1.dzone.com/storage/temp/17538546-screenshot-2024-02-28-at-95655-am.png" height="50%" width="50%" />

Refer to [Distance Measurement in Text Mining](ml/ml-concepts.md#distance-measurement-in-text-mining) to know more about Euclidean Distance, Cosine Distance & Dot Product

### Vector Index

Instead of checking distances between each vector in the database, we build index structures that narrow down the search space and improve lookup times.

A **vector index is a condensed form of raw vectors** that allow efficient, rapid searches. Indices are created by different algorithmic approaches. Primary categories of vector indexes,

- **Flat** (e.g. Brute Force)
- **Graph-based** (e.g., HNSW - Hierarchical Navigable Small Words)
- **Inverted** (e.g. IVF, IVF-PQ)

### Flat Index

With flat indices, search is exhaustive: it’s performed from the query vector across every single vector embedding and distances are calculated for each pair. Then, k number of embeddings are returned using k-nearest neighbors (kNN) search.

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*Po5wt0bU4V2KycKL0xA0nA.png" width="50%" height="50%" />
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*4-vyY60jeP2A8JoKtkH8dQ.png" width="50%" height="50%" />

### Graph Index

Graph indices use nodes and edges to construct a network-like structure. The nodes (or “vertices”) represent the vector embeddings while the edges represent the relationships between embeddings. The most common type of graph index is Hierarchical Navigable Small Words (HNSW).

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*4l7XPLWhrZ2dnQ_4I9IwOw.png" width="50%" height="50%" />
