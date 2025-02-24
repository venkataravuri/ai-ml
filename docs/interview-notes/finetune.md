#### What is Parameter-Efficient Fine-Tuning (PEFT)?

PEFT Finetuning is Parameter Efficient Fine Tuning, a set of fine-tuning techniques that allows you to fine-tune and train models much more efficiently than normal training.

PEFT techniques usually work by reducing the number of trainable parameters in a neural network. The most famous and in-use PEFT techniques are Prefix Tuning, P-tuning, LoRA, etc. LoRA is perhaps the most used one.

### Can you explain the differences between LoRA and QLoRA? How do you implement LoRA in a model?

**QLORA**

> BitsandBytes library takes care of the 4-bit quantization and the whole low-precision storage and high-precision compute part.

```python
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) # setup bits and bytes config

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model) # prepares the whole model for kbit training

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config) # Now you get a model ready for QLoRA training
```
        
### What is the role of Unsloth in accelerating fine-tuning processes? How does it improve memory efficiency and speed?

### How do you handle padding in sequences during training? ### Discuss strategies for effective padding management.

### What are system tokens in Llama Models?
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
- **<|begin_of_text|>**: This is equivalent to the BOS token
- **<|eot_id|>**: This signifies the end of the message in a turn.
- **<|start_header_id|>**{role}**<|end_header_id|>**: These tokens enclose the role for a particular message. The possible roles can be: system, user, assistant.
- **<|end_of_text|>**:	This is equivalent to the EOS token. On generating this token, Llama 3 will cease to generate more tokens

### What is the significance of setting a maximum sequence length? How does it affect model performance and memory usage?

### Explain Proximal Policy Optimization (PPO) and its application in fine-tuning LLMs? What are its advantages compared to other optimization algorithms?

### What is Direct Preference Optimization (DPO), and how does it differ from PPO How do you perform progress logging during model training?

### What is the purpose of the Accelerate library in training LLMs? How does it facilitate distributed training?
    
### Can you explain how to use the SFTTrainer class from Hugging Face's trl library? What are its key features?

### What are ommon pitfalls when fine-tuning LLMs?

- **Overfitting:** This occurs when a model learns the training data too well, leading to high accuracy on that dataset but poor generalization to new data.
- **Catastrophic Forgetting:** Description: When fine-tuning on a new task, the model may lose its ability to perform well on previously learned tasks.
- **Hyperparameter Misconfiguration:** Description: Incorrect settings for hyperparameters such as learning rate, batch size, and number of epochs can lead to suboptimal model performance.

### Which loss function is used in Llama 3.1 models?

Llama 3.1 models is primarily based on cross-entropy loss, which is standard for training language models.

This loss function measures the dissimilarity between the predicted probability distribution of the model and the actual distribution of the target labels.

While cross-entropy loss is used during training, various evaluation metrics (like accuracy, F1 score, etc.) are employed post-training to assess model performance on specific tasks

### While Fine-tuning LLMs using distributed training across multiple nodes and GPUs using PyTorch FSDP (Fully Sharded Data Parallel),
    Is it possible to use HuggingFace Transformers and SFT Trainer for distributed training?
    How data should be distributed? Is data should be split before running training?
    How data loader knows which records should be loaded? Each node will get different dataset or same dataset?

**Data Distribution:**
- The data is automatically sharded across nodes using the DistributedDataset class
- Each node gets a unique subset of the data based on its rank and world size
- The data is split at runtime, so you don't need to split it beforehand
- Each node processes different data to avoid redundancy

**Data Loading:**
- The dataloader knows which records to load through the rank-based sharding
- Each node's dataset is determined by: shard_size = total_data // world_size
- Data indices for each node: start_idx = rank * shard_size

**FSDP Configuration:**
- Uses mixed precision training (FP16) for efficiency
- Implements transformer-specific auto wrap policy
- Enables activation checkpointing to save memory
- Configures backward prefetching for better performance

**Distributed Training Setup:**
- Initializes distributed environment using NCCL backend
- Sets up proper device mapping for multi-GPU training
- Handles proper model distribution across GPUs using FSDP

```bash
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 --master_addr="master_ip" --master_port=29500 train.py
```

<details>

<summary>Show code sample - Training script across multiple nodes (Generated by Claude.ai)</summary>

```python
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from typing import Dict, List
def setup_distributed():
    """ Initialize distributed training environment """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        raise ValueError("RANK and WORLD_SIZE environment variables must be set")

class DistributedDataset(Dataset):
    def __init__(self, dataset, rank: int, world_size: int, tokenizer):
        """
        Initialize dataset with sharding for distributed training
        
        Args:
            dataset: HuggingFace dataset
            rank: Current process rank
            world_size: Total number of processes
            tokenizer: HuggingFace tokenizer
        """
        # Shard the dataset
        self.shard_size = len(dataset) // world_size
        start_idx = rank * self.shard_size
        end_idx = start_idx + self.shard_size if rank != world_size - 1 else len(dataset)
        
        self.data = dataset[start_idx:end_idx]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Assuming 'text' is the column name in your dataset
        encoded = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

def get_fsdp_config():
    """Configure FSDP settings"""
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    return {
        "mixed_precision": mixed_precision_policy,
        "auto_wrap_policy": transformer_auto_wrap_policy,
        "sharding_strategy": "FULL_SHARD",
        "cpu_offload": False,
        "backward_prefetch": "BACKWARD_PRE",
        "activation_checkpointing": True,
    }

def main():
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    
    # Initialize model and tokenizer
    model_name = "facebook/opt-350m"  # Example model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Wrap model with FSDP
    fsdp_config = get_fsdp_config()
    model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        **fsdp_config
    )
    
    # Load and prepare dataset
    dataset = load_dataset("your_dataset_name")
    train_dataset = DistributedDataset(
        dataset['train'],
        rank,
        world_size,
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=100,
        save_steps=1000,
        local_rank=local_rank,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save model (only on rank 0)
    if rank == 0:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
            torch.save(state_dict, "model_checkpoint.pt")

if __name__ == "__main__":
    main()

```
</details>

### While fine-tune LLMs using the SFT Trainer for mixed tasks (such as text generation, question answering, and summarization), how to pass multiple evaluation metrics task-wise to compute performance?

<details>
Set up the SFT Trainer in Hugging Face's transformers library to fine-tune a model for mixed tasks (text generation, question answering, and summarization) while specifying multiple evaluation metrics.
<summary>Show code sample (Generated by Claude.ai)</summary>
```python
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_metric
from typing import Dict, List, Union
import numpy as np

class MultiTaskTrainer(Trainer):
    """Custom Trainer class for handling multiple tasks"""
    def __init__(self, *args, **kwargs):
        # Extract task-specific metrics before passing remaining args to parent
        self.metrics = kwargs.pop('metrics', {})
        super().__init__(*args, **kwargs)

    def compute_metrics(self, eval_pred):
        """Compute metrics for all tasks"""
        predictions, labels = eval_pred
        # First element in labels is assumed to be task_ids
        task_ids = labels[:, 0].tolist()
        # Remove task_ids from labels
        labels = labels[:, 1:]
        
        # Convert logits to predictions
        predictions = np.argmax(predictions, axis=-1)
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Group predictions by task
        task_predictions: Dict[str, List] = {
            "text_generation": [],
            "question_answering": [],
            "summarization": []
        }
        task_labels: Dict[str, List] = {
            "text_generation": [],
            "question_answering": [],
            "summarization": []
        }
        
        # Map task IDs to task types
        task_map = {0: "text_generation", 1: "question_answering", 2: "summarization"}
        
        # Group predictions and labels by task
        for pred, label, task_id in zip(decoded_preds, decoded_labels, task_ids):
            task_type = task_map[task_id]
            task_predictions[task_type].append(pred)
            task_labels[task_type].append(label)
        
        # Compute metrics for each task
        results = {}
        for task_type in task_predictions:
            if len(task_predictions[task_type]) > 0:
                task_metric = self.metrics[task_type].compute(
                    predictions=task_predictions[task_type],
                    references=task_labels[task_type]
                )
                # Add task prefix to metric names
                results.update({
                    f"{task_type}_{k}": v 
                    for k, v in task_metric.items()
                })
        
        return results

class MultiTaskDataset:
    """Dataset class that handles multiple tasks"""
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Determine task type from dataset
        task_type = item['task_type']  # Assume dataset has this field
        task_id = {
            "text_generation": 0,
            "question_answering": 1,
            "summarization": 2
        }[task_type]
        
        # Encode input and output
        inputs = self.tokenizer(
            item['input_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        outputs = self.tokenizer(
            item['output_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Add task_id as first token of labels
        labels = outputs['input_ids'].clone()
        labels = torch.cat([torch.tensor([[task_id]]), labels], dim=1)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def main():
    # Load model and tokenizer
    model_name = "your-model-name"  # Replace with your model name
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load metrics
    metrics = {
        "text_generation": load_metric("bleu"),
        "question_answering": load_metric("squad"),
        "summarization": load_metric("rouge")
    }
    
    # Load dataset
    dataset = load_dataset("your-dataset-name")
    
    # Create multi-task datasets
    train_dataset = MultiTaskDataset(dataset['train'], tokenizer)
    eval_dataset = MultiTaskDataset(dataset['validation'], tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir='./logs',
    )
    
    # Initialize the custom trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        metrics=metrics
    )
    
    # Train and evaluate
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
```
</details>

### GPTQ vs. BitsandBytes (NF4)

NF4 is part of the bitsandbytes library,
- **Normalization**: Weights are normalized before quantization, allowing for efficient representation of common values.
- **4-bit Quantization**: Weights are quantized to 4 bits with evenly spaced levels based on normalized weights.
- **Dequantization During Computation**: Although weights are stored in 4-bit format, they are dequantized during computation to enhance performance.


GPTQ stands for Post-Training Quantization,
- **Dynamic Dequantization**: Weights are dynamically dequantized to float16 during inference, which improves performance while keeping memory requirements low.
- **Custom Kernels**: Utilizes specialized kernels for matrix-vector operations, resulting in faster processing speeds compared to other methods like bitsandbytes and GGML.
- **Performance Metrics**: In tests with models like Llama-7B, GPTQ showed lower perplexity (PPL) scores and higher token generation rates than both GGML and NF4


#### Skip Connections vs. Residual
there are two fundamental ways that one could use skip connections through different non-sequential layers:
a) **addition** as in residual architectures,
b) **concatenation** as in densely connected architectures.

<img src="https://theaisummer.com/static/8d19d048cd68d6dce362e025cf3b635a/1ac66/skip-connection.png"/>
<img src="https://theaisummer.com/static/b8156f7a258e0c46eb1e5e7b6bb591bf/ad12c/resnet-concatenation.png" />
