import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
model_id = "lexlms/legal-longformer-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Load the dataset
dataset = load_dataset("darrow-ai/USClassActions")

# Output CSV file for storing predictions
output_csv = "legal_longformer_large_predictions.csv"
fieldnames = ["id", "target_text", "verdict", "prediction"]

# Function to tokenize input and make predictions
def predict_outcome(target_text):
    inputs = tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=4096)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    return "Win" if prediction == 1 else "Lose"  # Adjust based on model labels (0 = Lose, 1 = Win)

# Process the dataset
with open(output_csv, mode="w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    for entry in dataset["train"]:
        target_text = entry["target_text"]
        verdict = entry["verdict"]

        # Generate prediction
        try:
            prediction = predict_outcome(target_text)
        except Exception as e:
            print(f"Error processing case ID {entry['id']}: {e}")
            prediction = "Error"

        # Write to output CSV
        writer.writerow({
            "id": entry["id"],
            "target_text": target_text,
            "verdict": verdict,
            "prediction": prediction,
        })

print(f"Predictions saved to {output_csv}")
