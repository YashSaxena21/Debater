import csv
from datasets import load_dataset
from transformers import pipeline
import torch
import optimum

# Load the model pipeline
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
text_gen_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Load the dataset
dataset = load_dataset("darrow-ai/USClassActions")

# Prepare output CSV
output_file = "llama_3.2_predictions.csv"
fieldnames = ["id", "target_text", "verdict", "llama_prediction", "llama_reasoning"]

with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

# Helper function to generate a prediction
def generate_prediction(prompt):
    outputs = text_gen_pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
    )
    return outputs[0]["generated_text"]

# Processing the dataset and collecting results
correct_predictions = 0
total = 0

for entry in dataset["train"]:
    target_text = entry["target_text"]
    original_prediction = entry["verdict"]

    # Create the prompt
    prompt = (
        f"Analyze the following legal complaint and predict the outcome (\"Win\" or \"Lose\"). "
        f"Briefly explain your reasoning. Give the output in the format "
        f"(Prediction: <Win/Lose>), Reasoning: <Brief explanation>.\n\n"
        f"Complaint:\n{target_text}\n\n"
        f"Response:\n"
    )

    # Skip overly long inputs
    tokenized_length = len(prompt.split())  # Approximation of token length
    if tokenized_length > 1500:
        print(f"Skipping entry {entry['id']} due to excessive input length.")
        continue

    # Generate prediction
    try:
        llm_response = generate_prediction(prompt)
        print(f"Response: {llm_response}")

        # Extract prediction and reasoning
        prediction_part = (
            llm_response.split("Reasoning:")[0]
            .replace("Prediction:", "")
            .strip()
        )
        reasoning_part = (
            llm_response.split("Reasoning:")[1].strip()
            if "Reasoning:" in llm_response
            else "N/A"
        )

        # Clean prediction
        lawllm_prediction = prediction_part.strip()

        # Update accuracy
        if lawllm_prediction.lower() == original_prediction.lower():
            correct_predictions += 1
    except Exception as e:
        print(f"Error processing entry {entry['id']}: {e}")
        lawllm_prediction = "Error"
        reasoning_part = str(e)

    total += 1

    # Write to CSV
    with open(output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(
            {
                "id": entry["id"],
                "target_text": target_text,
                "verdict": original_prediction,
                "llama_prediction": lawllm_prediction,
                "llama_reasoning": reasoning_part,
            }
        )

# Calculate and display accuracy
accuracy = (correct_predictions / total) * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.2f}%")
