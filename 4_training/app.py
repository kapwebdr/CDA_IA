from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json

# 1. Charger le modèle et le tokenizer
model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Préparer le dataset
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Question: {item['question']}\nRéponse: {item['answer']}"
        
        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 3. Créer le dataset
dataset = CustomDataset("dataset.json", tokenizer)

# 4. Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./trained_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=50,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    remove_unused_columns=False
)

# 5. Initialiser et lancer l'entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# 6. Sauvegarder le modèle
model_path = "./trained_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# 7. Fonction de test
def generate_text(prompt):
    inputs = tokenizer(f"Question: {prompt}", return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 8. Test du modèle
test_questions = [
    "Qui est le rival principal de Naruto ?",
    "Quel est le pouvoir principal de Luffy ?"
]

print("\nTests du modèle entraîné:")
for question in test_questions:
    print(f"\nQuestion: {question}")
    print(f"Réponse: {generate_text(question)}")
