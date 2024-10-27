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

LoRA, or Low-Rank Adaptation, is a technique that modifies the architecture of a pre-trained model by **introducing low-rank matrices** into the model's layers. LoRA adds trainable low-rank matrices to selected layers, allowing the model to adapt to new tasks without the need for extensive computational resources.

- Preservation of Pre-trained Knowledge: The original model retains its pre-trained weights, and only the low-rank matrices are trained, allowing for faster adaptation to new tasks.
- LoRA is an efficient way to fine-tune large models by introducing low-rank adaptations that require fewer resources while preserving the knowledge embedded in pre-trained weights.

#### How to Choose Layers for Adding Adapters?

 - If your task heavily relies on understanding context, prioritize attention layers.
 - Later layers: Task-specific adaptations

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
