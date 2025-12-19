import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

# Load tokenizer & model
model_path = "emotion_model_go/checkpoint****" # Replace **** with the appropriate checkpoint number
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Emotion labels + emojis
id2label = {
    0: ("sadness", "😢"),
    1: ("joy", "😊"),
    2: ("love", "❤️"),
    3: ("anger", "😡"),
    4: ("fear", "😱"),
    5: ("surprise", "😲"),
    6: ("neutral", "😐")
}



def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()

    return pred_id, probs.tolist()

def analyze_paragraph(paragraph):
    sentences = sent_tokenize(paragraph)
    results = []

    print("\n================= Emotion Analysis =================\n")

    for i, sent in enumerate(sentences, start=1):
        pred_id, probs = predict_emotion(sent)
        emotion, emoji = id2label[pred_id]

        print(f"[{i}] {sent}")
        print(f"   → Emotion: {emotion} {emoji}")
        print(f"   → Confidence: {max(probs)*100:.2f}%\n")

        results.append(emotion)

    # Summary
    print("=============== Summary ===============\n")
    counter = Counter(results)

    for emotion, count in counter.items():
        print(f"{emotion.capitalize():10} : {count} sentences")

    most_common = counter.most_common(1)[0][0]
    print(f"\nMost frequent emotion: **{most_common.upper()}**\n")


# MAIN
if __name__ == "__main__":
    print("Paste your paragraph below:")
    paragraph = input("> ")
    analyze_paragraph(paragraph)
