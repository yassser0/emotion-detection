# 🧠 Emotion Detection from Text (NLP – DistilBERT + GoEmotions)

This project implements a state-of-the-art emotion classification system using DistilBERT fine-tuned on a reduced version of Google’s GoEmotions dataset. Given any text or paragraph, the model predicts one of 7 emotions: Joy, Sadness, Love, Anger, Fear, Surprise, or Neutral.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yassser0/emotion-detection.git
   cd emotion-detection
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install torch transformers datasets scikit-learn nltk
   ```

## 🎯 Usage of the Model

1. **Train the model**: Run:
   ```bash
   python emotion_training.py
   ```
   This will:
   - Load GoEmotions
   - Tokenize text
   - Fine-tune DistilBERT
   - Save the model in `emotion_model_go/checkpoint-XXXX`. After this, change `emotion_model_go/checkpoint-XXXX` to the real checkpoint in `analyze_text.py` & `predict_emotion.py`.

2. **Run Emotion Prediction**:
   - For only a sentence:
     ```bash
     python predict_emotion.py
     ```
   - Or analyze a full paragraph:
     ```bash
     python analyze_text.py
     ```

## 🧪 Dataset

To view the project dataset (Google GoEmotions, 2021), run:
```bash
python data_set.py
```

## 📄 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



