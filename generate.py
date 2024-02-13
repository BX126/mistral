from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def generate_prompt_chat(input, output,):
    return f"""<s>[INST] 
{input}[/INST]
{output}</s>
"""

data = {"Input": "How do I make a cake?", "Output": "First, you need to mix the flour, sugar, and eggs together."}

encodeds = tokenizer(generate_prompt_chat(data["Input"],data["Output"] ), return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
