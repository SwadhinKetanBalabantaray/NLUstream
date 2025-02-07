# NLUstream
from nlustream import Pipeline
from nlustream.components import SpacyPreprocessor, IntentClassifier, EntityExtractor

pipeline = Pipeline([
    SpacyPreprocessor(model="en_core_web_sm"), # Example using Spacy
    IntentClassifier(model_path="intent_model.pkl"), # Your custom intent model
    EntityExtractor(model="ner_model.pkl") # Your custom NER model
])

text = "Book a flight to London."
result = pipeline.process(text)
print(result) # Print the extracted intents, entities, etc.
