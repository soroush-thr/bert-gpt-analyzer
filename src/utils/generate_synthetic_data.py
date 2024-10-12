import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_samples=1000):
    companies = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Facebook']
    sentiments = ['positive', 'negative', 'neutral']
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for _ in range(num_samples):
        company = np.random.choice(companies)
        sentiment = np.random.choice(sentiments)
        date = start_date + timedelta(days=np.random.randint(0, 365))
        
        if sentiment == 'positive':
            headline = f"{company} reports strong quarterly earnings, stock surges"
        elif sentiment == 'negative':
            headline = f"{company} faces regulatory challenges, shares drop"
        else:
            headline = f"{company} announces new product line, market response mixed"
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'company': company,
            'headline': headline
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/synthetic_financial_news.csv', index=False)
    print(f"Generated {num_samples} synthetic news items and saved to data/raw/synthetic_financial_news.csv")

if __name__ == "__main__":
    generate_synthetic_data()