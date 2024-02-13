import torch
import pickle
import transformers
from trl import SFTTrainer
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


# Load the dataset
dataset = pickle.load(open("/home/bingxuan/mistral/dataset_mistral.pkl", "rb"))

# Load model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


# Configure LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
modules = find_all_linear_names(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(
    f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")


# Train the model
tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()
trainer.save_model()
