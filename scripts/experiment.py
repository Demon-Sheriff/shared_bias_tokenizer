print("importing dependencies")
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
print("done imports")
print("import completed")
uniform_tokenizer = "./tokenizers/uniform_tokenizer.json"
base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

custom_uniform_hf_tokenizer = Tokenizer.from_file(uniform_tokenizer_path)
base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
base_tokenizer._tokenizer = custom_uniform_hf_tokenizer
base_tokenizer.pad_token = base_tokenizer.eos_token # this line

print("Loading Model : gpt-neo-125M")
model_uniform = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
print("Model Loading complete")
print("========================")
print("Loading SFT dataset")
ds = load_dataset("projectbaraat/hindi-instruct-dataset-v0.1", split="train[:5000]") # first 5k rows
print("dataset loading complete")
print("=========================")
dsT = ds

n = min(len(dsT['instruction']), len(dsT['output']))
texts = []
for i in range(n):
    if len(texts) >= 10000:
        break
    instr = dsT['instruction'][i][0]['content']
    resp = dsT['output'][i][0]['content']
    texts.append(
        f"Instruction: {instr}\nResponse: {resp}{base_tokenizer.eos_token}"
    )

hf_dataset = Dataset.from_dict({"text": texts[:100]})

def tokenize_function(examples):
    return base_tokenizer(examples["text"], truncation=True, max_length=128) # Adjust max_length

tokenized_datasets_uniform = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=base_tokenizer, mlm=False)
training_args_uniform = TrainingArguments(
    output_dir="./results_uniform",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs_uniform",
    logging_steps=20,        # log every 50 steps
    evaluation_strategy="no",
    eval_steps=200,          # evaluate every 200 steps
    fp16=True,               # mixed precision for GPU speed
    optim="adamw_torch_fused",
    report_to=["wandb"],    # use wandb if logged in
)

trainer_uniform = Trainer(
    model=model_uniform,
    args=training_args_uniform,
    train_dataset=tokenized_datasets_uniform,
    data_collator=data_collator,
)

    

