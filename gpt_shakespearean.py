from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd
import random
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model', type=str, default="gpt2")
parser.add_argument('--data', type=str, default="data/Shakespeare_data.csv")
parser.add_argument('--device', type=str, default="mps")
parser.add_argument('--num_epochs', type=int, default=3)
args = parser.parse_args()

# Load data and split into training and testing sets
data = pd.read_csv(args.data)
playerLines = list(data["PlayerLine"])

random.shuffle(playerLines)
train_data = playerLines[:8000]
test_data = playerLines[8000:]

with open('data/shakespeare_test_data.txt','w') as tfile:
	tfile.write('\n'.join(test_data))

with open('data/shakespeare_train_data.txt','w') as tfile:
	tfile.write('\n'.join(train_data))

# Set the device parameter
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
device = args.device

# Load the GPT-2 model and tokenizer
model_name = args.pretrained_model
config = GPT2Config.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Define the dataset for training
train_file = "data/shakespeare_train_data.txt"
test_file = "data/shakespeare_test_data.txt"

train_set = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
test_set = TextDataset(tokenizer=tokenizer, file_path=test_file, block_size=128)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=test_set,
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("gpt_shakespearean")
tokenizer.save_pretrained("gpt_shakespearean")
