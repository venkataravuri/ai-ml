# LLM Transformer Architecture

:star::star::star::star::star: **LLM Visualization**: https://bbycroft.net/llm


Most of LLMs in market are based variants of the original [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) paper that introduced transformers presented an encoder-decoder architecture.

Transformer models produce a probability distribution over all potential next words given an input string of text. 

- The original paper introduced encoder-decoder transformers architecture, which is more geared to tasks like translation.
- ChatGPT and other GPT-series models are decoder-only transformers.

Source: [](https://benlevinstein.substack.com/p/a-conceptual-guide-to-transformers)

The main architectural innovation of transformers was the introduction of attention heads.

### References

- [Transformers Explained Visually (Part 3): Multi-head Attention, deep dive](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)
- [A Gentle Introduction to Positional Encoding in Transformer Models, Part 1](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Explaining ChatGPT to Anyone in <20 Minutes](https://cameronrwolfe.substack.com/p/explaining-chatgpt-to-anyone-in-20)

### Rotary Position Embedding (RoPE)

LLaMA2 adopts Rotary Position Embedding (RoPE) in place of traditional absolute positional encoding. 

Read more about RoPE at, [Rotary Embedding](https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7)

## Self Attention

In self-attention, every word in sequence pays attention to every other word to understand context. Self attention allows the model to relate words each other.

Given a query, lookup for closest keys, return a weighted sum of associated values.

## Multi-head Attention

<img src="https://user-images.githubusercontent.com/16246821/79481335-f70d9400-802c-11ea-83f7-6f470fe46196.png" />

<img src="https://camo.githubusercontent.com/c0d5357cfe7029b123efe716a0c99d92d05956097c743449457fdf157c5ab3ca/68747470733a2f2f6d6368726f6d69616b2e6769746875622e696f2f61727469636c65732f323031372f5365702f31322f5472616e73666f726d65722d417474656e74696f6e2d69732d616c6c2d796f752d6e6565642f696d672f4d756c7469486561642e706e67" />

## Cross Attention

## Grouped Query Attention

GQA addresess memory bandwidth challenges during the autoregressive decoding of Transformer models. The primary issue stems from the need to load decoder weights and attention keys/values at each processing step, which consumes excessive memory.

Read more about MPA & GPA at, [Grouped Query Attention](https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7)

## Sliding Attention

## Flash Attention

**Transformers** are slow and memory-hungry on **long sequences**, since the time and memory complexity of self-attention are **quadratic in sequence length**.

Flash attention uses two techniques to speedup,

- **"tiling"** to "restructure the computation of attention" by splitting the input into blocks and performing softmax incrementally.
- **I/O aware implementation of attention**: Instead of storing the matrix for backpropagation, we simply recalculate it, which is faster than the I/Os.

<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F04f9b12d-eec9-4558-86ee-b23e03807935_1600x889.jpeg" width="70%" height="70%" />

<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3ec9eb47-c496-4a15-b8b9-ed091de6c06e_1932x680.png" width="70%" height="70%" />

Source: [FlashAttention challenges ML researchers to think about systems-level improvements](https://dailyink.substack.com/p/flashattention-challenges-ml-researchers) [Long-Sequence Attention with ⚡FlashAttention⚡](https://mlnotes.substack.com/p/long-sequence-attention-with-flashattention), 

### KV Caching

To understand KV caching, it’s important to know how the self-attention mechanism operates within transformers. In each self-attention layer, a given input sequence is projected into three separate matrices: queries (Q), keys (K), and values (V). These projections allow the model to determine which parts of the sequence to attend to by computing attention scores between the queries and keys, which are then applied to the values to yield the output.

Here's a breakdown of how KV caching fits into this process:

- **Initial Computation**: When generating the first few tokens in a sequence, the model computes the attention matrices—Q, K, and V—for all tokens processed so far. This involves full computation of attention for these initial tokens.
- **Caching of Keys and Values**: After computing K and V for each token, these matrices are stored in a KV cache. This cache holds the keys and values for each token in the sequence, indexed by their position.
- **Reuse in Subsequent Tokens**: For each new token generated (in sequential text generation or conversation), the model only computes the query (Q) for the current token. Instead of recalculating K and V for the entire sequence, it retrieves the cached K and V matrices from the previous steps.
- **Attention Calculation**: The attention is then computed using the new query (Q) and the cached keys (K) and values (V). This allows the model to attend to the entire history of tokens without recomputing each one from scratch.

<img src="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/kv-cache-optimization.png" height="70%" width="70%" />

<img src="https://martinlwx.github.io/img/kv_cache_4_steps.png" height="70%" width="70%" />

#### Handling Memory Limitations in KV Cache

One challenge with KV caching is that it requires memory for each token generated, which can be substantial for long sequences. Some approaches to manage this include:

<img src="https://pic4.zhimg.com/v2-6312af05f3f22e735bb6845e67d9b447_r.jpg" height="70%" width="70%" />

