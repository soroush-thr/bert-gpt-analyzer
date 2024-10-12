import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BERTSentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probabilities, dim=-1)
        return sentiment.item()

    def analyze_sentiment(self, df):
        df['sentiment'] = df['processed_text'].apply(self.predict_sentiment)
        return df