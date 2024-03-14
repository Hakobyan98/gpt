import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="gpt_shakespearean")
parser.add_argument('--length', type=int, default=500, help='the length of the text to generate')
args = parser.parse_args()


# Load fine-tuned model and tokenizer
model_name = args.model_name
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Generate Shakespearean-style text of desired length
desired_length = args.length
output = model.generate(max_length=desired_length, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
