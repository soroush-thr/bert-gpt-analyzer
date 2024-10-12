import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding

def create_training_data(labeled_data):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    for text, annotations in labeled_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)
    return doc_bin

def train_ner_model(train_data, output_dir, n_iter=100):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    for _, annotations in train_data:
        for _, _, label in annotations:
            ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print(f"Iteration {itn}, Losses: {losses}")

    nlp.to_disk(output_dir)

class FinancialNER:
    def __init__(self, model_path):
        self.nlp = spacy.load(model_path)

    def extract_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]