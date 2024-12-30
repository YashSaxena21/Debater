from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import re

dataset = load_dataset('darrow-ai/USClassActions')
print(dataset)
dataset = dataset['train']

model1 = AutoModelForCausalLM.from_pretrained("yashsaxena21/DPO_Legal_Llama-3.1-8b-Instruct")
tokenizer1 = AutoTokenizer.from_pretrained("yashsaxena21/DPO_Legal_Llama-3.1-8b-Instruct")
model2 = AutoModelForCausalLM.from_pretrained("yashsaxena21/DPO_Legal_Contradictions_Llama-3.1-8b-Instruct")
tokenizer2 = AutoTokenizer.from_pretrained("yashsaxena21/DPO_Legal_Contradictions_Llama-3.1-8b-Instruct")

def extract_response(text):
    # Define the regex pattern to match text after "Response:"
    pattern = r"Response:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)  # DOTALL allows matching across newlines
    if match:
        return match.group(1).strip()
    return None

alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}
"""

for i in dataset:
    query = i['target_text']
    verdict = i['verdict']

    # Prepare defender input
    defender_input = alpaca_prompt.format(
        "Given the following legal 'Query', generate a relevant passage that best supports the query by citing judgements in similar cases. Avoid generating a passage that contradicts the query. Don't repeat the prompt in the output, just provide the passage.",
        f"Query: {query}",
        ""
    )
    defender_input_tokens = tokenizer1.encode(defender_input, return_tensors="pt")
    max_new_tokens_defender = defender_input_tokens.size(1) + 512

    inputs1 = tokenizer1([defender_input], return_tensors="pt").to("cuda")
    outputs1 = model1.generate(**inputs1, max_new_tokens=max_new_tokens_defender, use_cache=True)
    defender_response = tokenizer1.batch_decode(outputs1)
    defender_response = extract_response(defender_response[0])

    # Prepare prosecutor input
    prosecutor_input = alpaca_prompt.format(
        "Given the following legal 'Query', generate a passage that contradicts the argument present in the query by citing judgements that support these contradictions. The passage should present a counterargument.",
        f"Query: {query}",
        ""
    )
    prosecutor_input_tokens = tokenizer2.encode(prosecutor_input, return_tensors="pt")
    max_new_tokens_prosecutor = prosecutor_input_tokens.size(1) + 512

    inputs2 = tokenizer2([prosecutor_input], return_tensors="pt").to("cuda")
    outputs2 = model2.generate(**inputs2, max_new_tokens=max_new_tokens_prosecutor, use_cache=True)
    prosecutor_response = tokenizer2.batch_decode(outputs2)
    prosecutor_response = extract_response(prosecutor_response[0])

    headers = ["Query", "Defender Response", "Prosecutor Response", "Verdict"]

    with open("Legal_responses.csv", mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow({
            "Query": query,
            "Defender Response": defender_response,
            "Prosecutor Response": prosecutor_response,
            "Verdict": verdict
        })
