import pandas as pd
from datasets import load_dataset


dataset = load_dataset("go_emotions")
train_df = pd.DataFrame(dataset["train"])
train_df.to_csv("emotion_train.csv", index=False)
print("CSV file saved!")


