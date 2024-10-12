from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPTReportGenerator:
    def __init__(self, model_name='gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate_report(self, prompt, max_length=200):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def summarize_sentiment(self, df):
        positive = df[df['sentiment'] == 1].shape[0]
        negative = df[df['sentiment'] == 0].shape[0]
        total = df.shape[0]
        
        prompt = f"Generate a brief market sentiment report based on the following data: {positive} positive sentiments, {negative} negative sentiments out of {total} total analyzed documents."
        return self.generate_report(prompt)