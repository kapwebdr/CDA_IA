from peft import get_peft_model, LoraConfig, TaskType  # PEFT permet d'utiliser des techniques comme LoRA pour fine-tuner efficacement les modèles.
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
import torch

# 1. Charger le modèle
model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. Charger et préparer le dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset_raw = json.load(f)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Formater le texte avec un format spécifique
        full_text = f"Question: {item['question']}\nRéponse: {item['answer']}"
        
        # Tokeniser avec padding à droite
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Préparer les entrées avec les labels
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze().clone()  # Important pour le calcul de la loss
        }

dataset = CustomDataset(dataset_raw, tokenizer)

# 3. Configuration LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 4. Appliquer LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    remove_unused_columns=False,
    optim="adamw_torch"
)

# 6. Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: dict((key, torch.stack([f[key] for f in data])) for key in data[0])
)

# 7. Lancer l'entraînement
trainer.train()

# 8. Sauvegarder le modèle
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# 9. Charger le modèle fine-tuné pour le tester
model = AutoModelForCausalLM.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

# 10. Fonction pour tester le modèle
def chat_with_model(question):
    """ Génère une réponse en fonction d'une question fournie """
    prompt = f"Question: {question}\nRéponse:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Génération de texte avec une longueur max de 50 tokens
    output = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# 11. Exemple de test
test_question = "Quel est le rival principal de Naruto ?"
print(chat_with_model(test_question))