import os

import tensorflow as tf
from datasets import Dataset, load_dataset
from transformers import TFGPT2LMHeadModel, \
    GPT2Tokenizer, TFTrainer, TFTrainingArguments, TFGPT2Model, AutoTokenizer
from transformers import TFGPT2LMHeadModel

from DataLoader import load_programs_to_dict

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset_dict = load_programs_to_dict(os.path.join("cleared_data", "metaset3.dev.jsonl"))
dataset = Dataset.from_dict(dataset_dict)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
text = "Replace me by any text you'd like."
tokenizer.add_special_tokens({
    "pad_token": "[PAD]"
})
model = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id)


# print(tokenizer.decode(output))
def encode(examples):
    return tokenizer(examples["text"], return_tensors="tf", truncation=True, padding=True, max_length=1500)

print(model.input_names)

tensor = []
for i in dataset:
    tensor.append(tokenizer.encode(i["text"], return_tensors="tf", truncation=True, padding=True, max_length=1000))

train_dataset = dataset.map(encode)
print(train_dataset[0])
print(text)

print(tokenizer.encode("Replace me by any text you'd like.", return_tensors="tf"))
train_dataset = tf.data.Dataset.from_tensor_slices(tensor)


training_args = TFTrainingArguments(
    output_dir="./gpt2-gerchef",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=32,  # batch size for training
    # per_device_eval_batch_size=64,  # batch size for evaluation
    # eval_steps=400,  # Number of update steps between two evaluations.
    save_steps=800,  # after # steps model is saved
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

dt = trainer.get_train_tfdataset()

trainer.train()
