import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import optimum

# Hugging Face login
# Replace 'your_hf_token' with your actual Hugging Face token or ensure you are already logged in
hugging_face_token = "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"  # Add your token here
login(hugging_face_token)

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load the dataset
dataset = load_dataset('darrow-ai/USClassActions')

# Prepare output CSV
output_file = "legal_Mistral_predictions.csv"
fieldnames = ["id", "target_text", "verdict", "mistral_prediction", "mistral_reasoning"]

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

# Helper function to generate a prediction
def generate_prediction(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_length = inputs.shape[1]  # Calculate the number of input tokens

    # Set max_new_tokens dynamically
    max_new_tokens = input_length + 512

    # Generate response
    outputs = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Processing the dataset and collecting results
correct_predictions = 0
total = 0

for entry in dataset['train']:
    target_text = entry['target_text']
    original_prediction = entry['verdict']

    # Create the prompt
    prompt = (
        f"Analyze the following legal complaint and predict the outcome (\"Win\" or \"Lose\"). "
        f"Briefly explain your reasoning. Give the output in the format "
        f"Prediction: <Win/Lose>, Reasoning: <Brief explanation>.\n\n"
        f"Complaint:\n{target_text}\n\n"
        f"Response:\n"
    )

    # Generate prediction
    try:
        llm_response = generate_prediction(prompt)
        print(llm_response)

        # Extract prediction and reasoning from response
        prediction_part = llm_response.split("Reasoning:")[0].strip()
        reasoning_part = llm_response.split("Reasoning:")[1].strip() if "Reasoning:" in llm_response else "N/A"

        # Clean prediction
        mistral_prediction = prediction_part.replace("Prediction:", "").strip()

        # Update accuracy
        if mistral_prediction.lower() == original_prediction.lower():
            correct_predictions += 1
    except Exception as e:
        print(f"Error processing entry {entry['id']}: {e}")
        mistral_prediction = "Error"
        reasoning_part = str(e)

    total += 1

    # Write to CSV
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            "id": entry["id"],
            "target_text": target_text,
            "verdict": original_prediction,
            "mistral_prediction": mistral_prediction,
            "mistral_reasoning": reasoning_part
        })

# Calculate and display accuracy
accuracy = (correct_predictions / total) * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
