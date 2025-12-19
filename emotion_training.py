from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# 1. Load GoEmotions dataset
# ---------------------------
print("➡️ Loading GoEmotions dataset...")
dataset = load_dataset("go_emotions")
print(dataset)

# GoEmotions label names
id2label_go = dataset["train"].features["labels"].feature.names

# ---------------------------
# 2. Map GoEmotions → 7 emotions
# ---------------------------

# Our 7 target emotions
label2id = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
    "neutral": 6
}

id2label = {v: k for k, v in label2id.items()}

# Map 27 GoEmotions labels to 7 classes
emotion_map = {
    "neutral": "neutral",
    "sadness": "sadness",
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "love": "love",
    "caring": "love",
    "desire": "love",
    "anger": "anger",
    "annoyance": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "realization": "surprise",
}

default_emotion = "neutral"

def map_emotion(labels):
    """
    Convert the multi-label emotion list into ONE dominant emotion.
    """
    emo_names = [id2label_go[i] for i in labels]

    for e in emo_names:
        if e in emotion_map:
            return emotion_map[e]
    
    return default_emotion

# Apply emotion mapping to dataset
dataset = dataset.map(lambda x: {"emotion": map_emotion(x["labels"])})

# Convert emotion string → numeric ID
dataset = dataset.map(lambda x: {"label": label2id[x["emotion"]]})

print("➡️ Sample mapped emotion:", dataset["train"][0]["text"], "→", dataset["train"][0]["emotion"])

# ---------------------------
# 3. Tokenization
# ---------------------------
print("\n➡️ Tokenizing dataset...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------
# 4. Load the model
# ---------------------------
print("\n➡️ Loading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=7,
    id2label=id2label,
    label2id=label2id
)

# ---------------------------
# 5. Metrics
# ---------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# ---------------------------
# 6. Training configuration
# ---------------------------
training_args = TrainingArguments(
    output_dir="./emotion_model_go",
    eval_strategy="epoch",
    save_strategy="epoch",         
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True     
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

# ---------------------------
# 7. Train!
# ---------------------------
print("\n🚀 Starting training...\n")
trainer.train()

print("\n✅ Training finished successfully!")
