# bert-gpt-analyzer

An advanced NLP pipeline using BERT, GPT, and custom NER models for market sentiment analysis, automated report generation, and financial document processing.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage Guide](#usage-guide)
  - [Data Preparation](#1-data-preparation)
  - [Data Preprocessing](#2-data-preprocessing)
  - [Sentiment Analysis](#3-sentiment-analysis)
  - [Report Generation](#4-report-generation)
  - [Named Entity Recognition (NER)](#5-named-entity-recognition-ner)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure

```
bert-gpt-analyzer/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── sentiment_analysis/
│   ├── report_generation/
│   └── ner/
├── src/
│   ├── data_processing/
│   │   └── preprocess.py
│   ├── sentiment_analysis/
│   │   └── bert_sentiment.py
│   ├── report_generation/
│   │   └── gpt_report.py
│   ├── ner/
│   │   └── financial_ner.py
│   └── utils/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/soroush-thr/bert-gpt-analyzer.git
   cd bert-gpt-analyzer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with PyTorch installation, visit https://pytorch.org/get-started/locally/ and follow the instructions for your specific system configuration.

## Usage Guide

### 1. Data Preparation

Place your raw financial data in the `data/raw/` directory.

### 2. Data Preprocessing

Use the `preprocess.py` script to clean and tokenize your data:

```python
from src.data_processing.preprocess import prepare_data

processed_data = prepare_data('data/raw/your_data.csv')
processed_data.to_csv('data/processed/processed_data.csv', index=False)
```

### 3. Sentiment Analysis

Use the BERT-based sentiment analysis model:

```python
from src.sentiment_analysis.bert_sentiment import BERTSentimentAnalyzer
import pandas as pd

# Load processed data
data = pd.read_csv('data/processed/processed_data.csv')

# Initialize and use the sentiment analyzer
analyzer = BERTSentimentAnalyzer()
data_with_sentiment = analyzer.analyze_sentiment(data)

# Save results
data_with_sentiment.to_csv('data/processed/data_with_sentiment.csv', index=False)
```

### 4. Report Generation

Generate reports based on sentiment analysis results:

```python
from src.report_generation.gpt_report import GPTReportGenerator
import pandas as pd

# Load data with sentiment
data = pd.read_csv('data/processed/data_with_sentiment.csv')

# Initialize and use the report generator
generator = GPTReportGenerator()
report = generator.summarize_sentiment(data)

print(report)
```

### 5. Named Entity Recognition (NER)

Train and use a custom NER model for financial document processing:

```python
from src.ner.financial_ner import create_training_data, train_ner_model, FinancialNER

# Prepare your labeled data
labeled_data = [
    ("Apple Inc. reported revenue of $100 billion in Q4 2023", [(0, 10, "ORG"), (31, 43, "MONEY"), (47, 55, "DATE")])
    # Add more labeled examples...
]

# Create training data
train_data = create_training_data(labeled_data)

# Train the model
train_ner_model(train_data, 'models/ner/financial_ner_model')

# Use the trained model
ner_model = FinancialNER('models/ner/financial_ner_model')
entities = ner_model.extract_entities("Microsoft's stock price reached $300 on January 15, 2024")
print(entities)
```

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Soroush Taheri - soroush.thr@gmail.com

Project Link: [https://github.com/soroush-thr/bert-gpt-analyzer](https://github.com/soroush-thr/bert-gpt-analyzer)