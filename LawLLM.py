import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "AdaptLLM/law-LLM"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load the dataset
dataset = load_dataset('darrow-ai/USClassActions')

# Prepare output CSV
output_file = "lawllm_predictions.csv"
fieldnames = ["id", "target_text", "verdict", "lawllm_prediction"]

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

# Helper function to generate a prediction
def generate_prediction(prompt):
    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_length = input_ids.shape[1]  # Number of tokens in the input

    # Set max_new_tokens to input_tokens + 512
    max_new_tokens = input_length + 512

    # Generate response with dynamic max_new_tokens
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Processing the dataset and collecting results
correct_predictions = 0
total = 0

for entry in dataset['train']:
    target_text = entry['target_text']
    original_prediction = entry['verdict']

    # Tokenize the target text to check its length
    tokenized_length = len(tokenizer.tokenize(target_text))
    if tokenized_length > 1500:
        print(f"Skipping entry {entry['id']} due to excessive token length ({tokenized_length} tokens).")
        continue

    # Create the prompt
    prompt = (
        f"Analyze the following legal complaint along with the defender and prosecutor responses, and predict the outcome (\"Win\" or \"Lose\") for the defendant.\n\n"
        f"Complaint:\n{target_text}\n\n"
        f"Prediction:\n"
    )

    # Generate prediction
    try:
        llm_response = generate_prediction(prompt)
        
        # Extract prediction from response
        lawllm_prediction = llm_response.replace("Prediction:", "").strip()

        # Update accuracy
        if lawllm_prediction.lower() == original_prediction.lower():
            correct_predictions += 1
    except Exception as e:
        print(f"Error processing entry {entry['id']}: {e}")
        lawllm_prediction = "Error"

    total += 1

    # Write to CSV
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            "id": entry["id"],
            "target_text": target_text,
            "verdict": original_prediction,
            "lawllm_prediction": lawllm_prediction
        })

# Calculate and display accuracy
accuracy = (correct_predictions / total) * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
