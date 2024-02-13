import json
import re
import pickle
from datasets import Dataset, load_dataset


data = json.load(
    open("/home/bingxuan/mistral/generated_fine_tuned_scrape.json", "r"))
n = len(data)
train, validation, test = data[:int(
    n*0.8)], data[int(n*0.8):int(n*0.9)], data[int(n*0.9):]


def generate_training_prompt_chat(input, output,):
    return f"""<s>[INST] 
{input}[/INST]
{output}</s>
"""

def process_data(data):
    dataset = []
    for d in data:
        input = d["Input"]
        output = d["Output"]
        prompt = generate_training_prompt_chat(input, output)
        dataset.append({"prompt": prompt, "input": input, "output": output})
    return dataset


train_dataset = process_data(train)
validation_dataset = process_data(validation)
test_dataset = process_data(test)

train_dataset = Dataset.from_list(train_dataset)
validation_dataset = Dataset.from_list(validation_dataset)
test_dataset = Dataset.from_list(test_dataset)

dataset = {"train": train_dataset,
           "validation": validation_dataset, "test": test_dataset}

print(dataset)

with open('dataset_mistral.pkl', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
