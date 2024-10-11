import ollama
import pandas as pd
import json
from tqdm import tqdm
import arabic_reshaper
from bidi.algorithm import get_display
from utils import iconqa_doc_to_text

def display_ar(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def generate(text):
    response = ollama.chat(model="qwen2.5:3b-instruct-fp16", messages=[
        {
            'role': 'user',
            'content': text
        }
    ], options=ollama.Options(temperature=1))
    return response['message']['content']

def evaluate_answer(pred_answer, true_answer, question):
    prompt = f"""
I will give you a question, a predicted answer, and a true answer. Your task is to determine if the predicted answer is correct based on the true answer. 
Please respond with 1 if the predicted answer matches the true answer, or 0 if it doesn't. Here's an example:

Example:
Question: What is the capital of France?
Predicted Answer: Paris
True Answer: Paris
Evaluation: 1

Now, please evaluate the following:
Question: {question}
Predicted Answer: {pred_answer}
True Answer: {true_answer}
Evaluation: """
    response = generate(prompt)
    return response.strip()

# Load the data
path = "results_silma9b.json"
with open(path, "r") as f:
    data = json.load(f)

report = []
for i in tqdm(range(len(data))):
    row = data[i]
    d = {}
    d['index'] = row['index']
    d['question'] = row['question']
    d['pred_answer'] = row['pred_answer']
    d['true_answer'] = row['answer']
    d['evaluation'] = evaluate_answer(d['pred_answer'], d['true_answer'], d['question'])
    report.append(d)

correct_count = sum(1 for item in report if item['evaluation'].strip() == '1')
total_count = len(report)
accuracy = correct_count / total_count

with open("results_silma9b.json", "w", encoding="utf-8") as f:
    json.dump({
        'results': report,
        'accuracy': accuracy
    }, f, ensure_ascii=False, indent=2)

print(f"Evaluation complete. Accuracy: {accuracy:.2%}")