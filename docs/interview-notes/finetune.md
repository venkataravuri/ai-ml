
### While fine-tune LLMs using the SFT Trainer for mixed tasks (such as text generation, question answering, and summarization), how to pass multiple evaluation metrics task-wise to compute performance?

<details>
<summary>Code sample</summary>

Below is a simplified example of how you might set up the SFT Trainer in Hugging Face's transformers library to fine-tune a model for mixed tasks (text generation, question answering, and summarization) while specifying multiple evaluation metrics. Set `task_type` dynamically based on which dataset or evaluation you're currently processing.

```python
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_metric

# Load your model and tokenizer
model_name = "your-model-name"  # Replace with your model name
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your dataset
# Assume you have a dataset that includes fields for different tasks
dataset = load_dataset("your-dataset-name")  # Replace with your dataset

# Define metrics for each task
metrics = {
    "text_generation": load_metric("bleu"),
    "question_answering": load_metric("squad"),
    "summarization": load_metric("rouge"),
}

# Define a compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Choose the metric based on the task type
    if task_type == "text_generation":
        return metrics["text_generation"].compute(predictions=decoded_preds, references=decoded_labels)
    elif task_type == "question_answering":
        return metrics["question_answering"].compute(predictions=decoded_preds, references=decoded_labels)
    elif task_type == "summarization":
        return metrics["summarization"].compute(predictions=decoded_preds, references=decoded_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
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
