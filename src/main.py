import os
import argparse
import pandas as pd
from src.data_processing.preprocess import prepare_data
from src.sentiment_analysis.bert_sentiment import BERTSentimentAnalyzer
from src.report_generation.gpt_report import GPTReportGenerator
from src.ner.financial_ner import create_training_data, train_ner_model, FinancialNER

def main(raw_file_name):
    # Ensure the necessary directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/ner', exist_ok=True)

    raw_file_path = os.path.join('data', 'raw', raw_file_name)
    
    # Step 1: Data Preprocessing
    print("Step 1: Data Preprocessing")
    processed_data = prepare_data(raw_file_path)
    processed_file_path = os.path.join('data', 'processed', f'processed_{raw_file_name}')
    processed_data.to_csv(processed_file_path, index=False)
    print(f"Processed data saved to {processed_file_path}")

    # Step 2: Sentiment Analysis
    print("\nStep 2: Sentiment Analysis")
    analyzer = BERTSentimentAnalyzer()
    data_with_sentiment = analyzer.analyze_sentiment(processed_data)
    sentiment_file_path = os.path.join('data', 'processed', f'sentiment_{raw_file_name}')
    data_with_sentiment.to_csv(sentiment_file_path, index=False)
    print(f"Sentiment analysis results saved to {sentiment_file_path}")

    # Step 3: Report Generation
    print("\nStep 3: Report Generation")
    generator = GPTReportGenerator()
    report = generator.summarize_sentiment(data_with_sentiment)
    report_file_path = os.path.join('data', 'processed', 'sentiment_report.txt')
    with open(report_file_path, 'w') as f:
        f.write(report)
    print(f"Generated report saved to {report_file_path}")
    print("\nReport Summary:")
    print(report[:500] + "..." if len(report) > 500 else report)

    # Step 4: Named Entity Recognition (NER)
    print("\nStep 4: Named Entity Recognition (NER)")
    # Simplified labeling for demonstration purposes
    labeled_data = [
        (row['headline'], [(row['headline'].index(row['company']), row['headline'].index(row['company']) + len(row['company']), "ORG")])
        for _, row in data_with_sentiment.iterrows() if row['company'] in row['headline']
    ]
    train_data = create_training_data(labeled_data)
    ner_model_path = os.path.join('models', 'ner', 'financial_ner_model')
    train_ner_model(train_data, ner_model_path)
    ner_model = FinancialNER(ner_model_path)
    
    # Extract entities from a sample headline
    sample_headline = data_with_sentiment['headline'].iloc[0]
    entities = ner_model.extract_entities(sample_headline)
    print(f"\nEntities extracted from sample headline:")
    print(f"Headline: {sample_headline}")
    print(f"Entities: {entities}")

    print("\nAnalysis complete. Check the 'data/processed' directory for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze financial news data")
    parser.add_argument("raw_file_name", help="Name of the raw data file in the data/raw directory")
    args = parser.parse_args()
    
    main(args.raw_file_name)
