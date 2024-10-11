import torch
from transformers import BertForSequenceClassification, BertTokenizer

import sys


# Now you can import functions from file_a.py
from models.custom_model import CustomEmailClassifier


from utils.preprocessor import preprocess_email
from utils.response_generator import generate_response
from sklearn.feature_extraction.text import CountVectorizer

class Chatbot:
    def __init__(self):
        # Load pretrained model
        self.pretrained_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.pretrained_model.load_state_dict(torch.load('models/pretrained_model.pth'))
        self.pretrained_model.eval()
        self.tokenizer = torch.load('models/pretrained_tokenizer.pkl')
        self.pretrained_label_encoder = torch.load('models/pretrained_label_encoder.pkl')

        # Load custom model
        input_size = 84 # This should match the max_features used in CountVectorizer
        hidden_size = 100
        num_classes = 3
        self.custom_model = CustomEmailClassifier(input_size, hidden_size, num_classes)
        self.custom_model.load_state_dict(torch.load('models/custom_model.pth'))
        self.custom_model.eval()
        self.vectorizer = torch.load('models/custom_vectorizer.pkl')
        self.custom_label_encoder = torch.load('models/custom_label_encoder.pkl')

    def classify_email_pretrained(self, email_content):
        preprocessed_email = preprocess_email(email_content)
        inputs = self.tokenizer(preprocessed_email, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.pretrained_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return self.pretrained_label_encoder.inverse_transform([predicted_class])[0]

    def classify_email_custom(self, email_content):
        preprocessed_email = preprocess_email(email_content)
        vector = self.vectorizer.transform([preprocessed_email]).toarray()
        with torch.no_grad():
            outputs = self.custom_model(torch.FloatTensor(vector))
        predicted_class = torch.argmax(outputs, dim=1).item()
        return self.custom_label_encoder.inverse_transform([predicted_class])[0]

    def generate_response(self, email_content):
        pretrained_category = self.classify_email_pretrained(email_content)
        custom_category = self.classify_email_custom(email_content)
        
        # We can implement a voting system or choose one of the models
        # For now, I am choosing the custom category model
        category = custom_category
        
        return generate_response(category, email_content)