import os

from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
import torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print(torch.cuda.is_available())
torch.cuda.set_device(0)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
print(train_dataset.examples[0])
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(torch.cuda.current_device())

training_args = TrainingArguments(
    output_dir="./gpt2-gerchef",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=32,  # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps=400,  # Number of update steps between two evaluations.
    save_steps=800,  # after # steps model is saved
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    prediction_loss_only=True,
)

trainer.train()
