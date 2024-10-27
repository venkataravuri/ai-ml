# Transformer Architecture

All most all LLMs in market are based variants of the original [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) paper that introduced transformers presented an encoder-decoder architecture.

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

# Finetune LLMs

## Concepts

### Prompt Tuning

#### Hard Prompting (Prompt Engineering)

Prompt engineering is a process that allows to engineer guidelines for a pre-trained model to implement a narrow task. A human engineer's instructions are fed to an LLM for it to accomplish a specific task. These instructions are called hard prompts.

For example, suppose we are interested in translating an English sentence into German. We can ask the model in various different ways, as illustrated below.
An example of hard prompt tuning, that is, rearranging the input to get better outputs.

```
1) Translate the English sentence '{English Sentence}' into German language '{German Translation}'
2) English: '{English Sentence}' | German: '{German Translation}'
3) From English to German: '{English Sentence}' -> '{German Translation}'
```

Now, this concept illustrated above is referred to as **hard prompt tuning** since we directly change the discrete input tokens, which are not differentiable.

[Source](https://magazine.sebastianraschka.com/p/understanding-parameter-efficient) 

#### Soft Prompting

In contrast to hard prompt tuning, soft prompt tuning (Lester et al. 2021) concatenates the embeddings of the input tokens with a trainable tensor that can be optimized via backpropagation to improve the modeling performance on a target task.

- Prompt tuning (different from prompting) appends a tensor to the embedded inputs of a pretrained LLM.
- The tensor is then tuned to optimize a loss function for the finetuning task and data while all other parameters in the LLM remain frozen.

Soft prompt tuning is significantly more parameter-efficient than full-finetuning.

### Prefix Tuning

Prefix tuning is to add trainable tensors to each transformer block instead of only the input embeddings, as in soft prompt tuning.

## In-context Learning vs Instruction Fine-tuning

In-context learning is a technique that leverages the LLM’s ability to learn from the context of the input. By providing a few prompt-completion examples before the actual query, the LLM can infer the task and the desired output format from the examples. In-context learning does not require any additional training of the model, but it relies on the model’s pre-trained knowledge and reasoning skills.

Instruction fine-tuning is a strategic extension of the traditional fine-tuning approach. Model is trained on examples of instructions and how the LLM should respond to those instructions.

## Parameter-Efficient Finetuning Methods

The main idea behind prompt tuning, and parameter-efficient finetuning methods in general, is to add a small number of new parameters
to a pretrained LLM and only finetune the newly added parameters to make the LLM perform better on,
- (a) a target dataset (for example, a domain-specific dataset like medical or legal documents)
- and (b) a target task (for example, sentiment classification).

## LoRA

## Quantization

- [Ultimate Guide to Fine-Tuning in PyTorch : Part 1 — Pre-trained Model and Its Configuration](https://rumn.medium.com/part-1-ultimate-guide-to-fine-tuning-in-pytorch-pre-trained-model-and-its-configuration-8990194b71e)

### Finetune Llama 2

- [LLaMA-2 from the Ground Up](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)
- [Fine-Tuning Llama-2 LLM on Google Colab: A Step-by-Step Guide.](https://medium.com/@csakash03/fine-tuning-llama-2-llm-on-google-colab-a-step-by-step-guide-cf7bb367e790)
- [How to Fine-Tune an LLM Part 2: Instruction Tuning Llama 2](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-2-Instruction-Tuning-Llama-2--Vmlldzo1NjY0MjE1)
- [Multiple tasks for one fine-tuned LLM](https://discuss.huggingface.co/t/multiple-tasks-for-one-fine-tuned-llm/31262/3)
- [Fine-tune Llama 2 with Limited Resources](https://www.union.ai/blog-post/fine-tune-llama-2-with-limited-resources)
- [Llama_2_7b_chat_hf_sharded_bf16_INFERENCE.ipynb](https://colab.research.google.com/drive/1zxwaTSvd6PSHbtyaoa7tfedAS31j_N6m)
- [fLlama_bnb_Inference.ipynb](https://colab.research.google.com/drive/1Ow5cQ0JNv-vXsT-apCceH6Na3b4L7JyW?usp=sharing#scrollTo=tMmDSVVaIfPF)
- []()


### Prompt to Generate Fine-tune Code

You are an AI and ML expert and developer, your job is to generate code for following use case,
Use Case: Finetune an open-source large language models (LLMs) such as latest Llama3 chat model for multi tasks such as Classification, Chatbot, Question Answering, Multi-choice question and answering and more. The model should be finetuned using below mentioned finetune datasets.

Below are few finetuning guidelines,
1. Use relevant HuggingFace libraries for finetuning.
2. The foundation LLM model and datasets should be downloaded from Huggingface, sometimes downloading models require Huggingface login and provide consent.
3. Employ finetuning techniques such as PEFT, LORA, QLORA and any modern techniques.
4. Use wandb.ai weights and bias tool for reporting progress and metrics
5. Finetune model for 2 epochs, use learning rate scheduler
6. Split data into training and validation set with 80% used for training and rest for validation. Validate finetuned model with validation dataset
7. Save & upload finetuned model to Huggingface
8. Use FSDP (Fully Sharded Data Parallel) to fine tune model. Assume that there are 3 nodes each having 4 A100 GPUs. The code should support distributed training across multiple nodes and gpus.
9. To stop retrain again due to crashes, do frequent check pointing.
10. Use model optimization techniques such as torch.compile, mixed precision or relevant techniques
11. Use Huggingface accelerate library
12. Finetune is planned to be executed on GPU renting services such as runpod.ai. Generate code that can be executed on such GPU renting services.
13. Compute evalution metrics and capture them.
14. Generate code as python project, separate behaviors into different modules so that it can be parameterized and packaged.
    
Finetune Datasets:
Below 3 Huggingface datasets should be used for finetuning,

1A) Dataset Name: infinite-dataset-hub/TextClaimsDataset
1B) Description: The 'TextClaimsDataset' is a curated collection of insurance claim descriptions where each text snippet is labeled according to its relevance to actual insurance claims. The dataset aims to assist machine learning practitioners in training models to classify texts as either 'Claim' or 'Not a Claim'. This classification can be pivotal for fraud detection systems in the insurance industry, helping to identify potential fraudulent claims from legitimate ones.
1C) Supported Tasks: Classification

2A) Dataset Name: PolyAI/banking77
2B) Dataset composed of online banking queries annotated with their corresponding intents. BANKING77 dataset provides a very fine-grained set of intents in a banking domain. It comprises 13,083 customer service queries labeled with 77 intents. It focuses on fine-grained single-domain intent detection.
2C) Supported Tasks: Intent classification, intent detection

3A) tau/commonsense_qa
3B) CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . It contains 12,102 questions with one correct answer and four distractor answers. The dataset is provided in two major training/validation/testing set splits: "Random split" which is the main evaluation split, and "Question token split"
3C) Supported Tasks: multiple-choice question answering
