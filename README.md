🧠Emotion Detection from Text (NLP – DistilBERT + GoEmotions) 
This project implements a state-of-the-art emotion classification system using DistilBERT fine-tuned on a reduced version of Google’s GoEmotions dataset.
Given any text or paragraph, the model predicts one of 7 emotions (Joy,Sadness,Love,Anger,Fear,Surprise or Neutral)

📦 Installation 
1. Clone the repository
   git clone https://github.com/yassser0/emotion-detection.git
   cd emotion-detection
2. Create a virtual environment
   python -m venv venv
   venv/bin/activate
3. Install dependencies
   pip install torch transformers datasets scikit-learn nltk

🎯 usage of the Model
1.train the model Run: 
    python emotion_training.py
This will Load GoEmotions Tokenize text Fine-tune DistilBERT Save the model in emotion_model_go/checkpoint-XXXX after this change  emotion_model_go/checkpoint-XXXX to the real checkpoint in analyze_text.py & predict_emotion.py
2.Run Emotion Prediction
for only a sentence :
      python predict_emotion.py
or Analyze a Full Paragraph : 
      python analyze_text.py
🧪 Dataset 
to view the project dataset Google GoEmotions (2021) run this: 
      python data_set.py
 


