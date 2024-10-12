import pandas as pd
from transformers import AutoTokenizer

def load_financial_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    # Add any text cleaning steps here
    return text.lower()

def tokenize_text(text, model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

def prepare_data(file_path, model_name='bert-base-uncased'):
    df = load_financial_data(file_path)
    
    # Check if 'headline' column exists, if not, use 'text' column
    text_column = 'headline' if 'headline' in df.columns else 'text'
    
    df['processed_text'] = df[text_column].apply(preprocess_text)
    df['tokenized'] = df['processed_text'].apply(lambda x: tokenize_text(x, model_name))
    return df
