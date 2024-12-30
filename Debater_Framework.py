import csv
from datasets import load_dataset
import transformers
import torch
from huggingface_hub import login

# Log in to Hugging Face
login(token="")  # Replace with your Hugging Face API token

# Model setup
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Load the dataset
dataset = load_dataset("darrow-ai/USClassActions")

# Input CSV file containing defender and prosecutor responses
input_csv = "Legal_responses.csv"  # Update with your input file path

# Output CSV file for storing predictions
output_csv = "debater_predictions.csv"
fieldnames = ["id", "target_text", "verdict", "Defender Response", "Prosecutor Response", "prediction"]

# Function to generate predictions
def generate_prediction(prompt):
    input_tokens = len(pipeline.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
    max_new_tokens = input_tokens + 512  # Adjust based on input tokens
    outputs = pipeline(prompt, max_new_tokens=max_new_tokens)
    return outputs[0]["generated_text"].strip()

# Process the dataset
with open(input_csv, mode="r", encoding="utf-8") as input_file, open(output_csv, mode="w", newline="", encoding="utf-8") as output_file:
    reader = csv.DictReader(input_file)
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    for entry, row in zip(dataset["train"], reader):
        target_text = entry["target_text"]
        verdict = entry["verdict"]
        defender_response = row["Defender Response"]
        prosecutor_response = row["Prosecutor Response"]

        # Construct the input prompt
        prompt = (
            f"Analyze the following legal complaint along with the plantiff's and defender's responses, and predict the outcome (\"Win\" or \"Lose\") for the plantiff.\n\n"
            f"Complaint:\n{target_text}\n\n"
            f"Plantiff's Response:\n{defender_response}\n\n"
            f"Defender's Response:\n{prosecutor_response}\n\n"
            f"Prediction:\n"
        )

        # Generate prediction
        try:
            prediction = generate_prediction(prompt)
        except Exception as e:
            print(f"Error processing case ID {entry['id']}: {e}")
            prediction = "Error"

        # Write to output CSV
        writer.writerow({
            "id": entry["id"],
            "target_text": target_text,
            "verdict": verdict,
            "Defender Response": defender_response,
            "Prosecutor Response": prosecutor_response,
            "prediction": prediction,
        })

print(f"Predictions saved to {output_csv}")
