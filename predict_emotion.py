import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

print("Running script from:", os.path.abspath(__file__))

model_path = "emotion_model_go/checkpoint****"  # Replace **** with the appropriate checkpoint number   
print("Loading model from:", model_path)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
    6: "neutral"
}

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    emotion_id = logits.argmax(dim=1).item()
    return id2label[emotion_id]

print("\n🔮 Emotion predictor is ready!\n")

while True:
    txt = input("Enter a sentence: ")
    if txt.lower() == "quit":
        break
    print(" → Predicted emotion:", predict_emotion(txt), "\n")
