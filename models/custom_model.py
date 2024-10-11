import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        return self.texts[item], self.labels[item]

class CustomEmailClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomEmailClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_custom_model():
    # Load data
    df = pd.read_csv('data/email_dataset.csv')
    texts = df['text'].tolist()
    labels = df['category'].tolist()

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Vectorize texts
    vectorizer = CountVectorizer(max_features=5000)
    train_vectors = vectorizer.fit_transform(train_texts).toarray()
    val_vectors = vectorizer.transform(val_texts).toarray()

    # Prepare datasets
    train_dataset = EmailDataset(torch.FloatTensor(train_vectors), torch.LongTensor(train_labels))
    val_dataset = EmailDataset(torch.FloatTensor(val_vectors), torch.LongTensor(val_labels))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    input_size = train_vectors.shape[1]
    hidden_size = 100
    num_classes = len(le.classes_)
    model = CustomEmailClassifier(input_size, hidden_size, num_classes)

    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {100 * correct / total:.2f}%")

    # Save the model, label encoder, and vectorizer
    torch.save(model.state_dict(), 'models/custom_model.pth')
    torch.save(le, 'models/custom_label_encoder.pkl')
    torch.save(vectorizer, 'models/custom_vectorizer.pkl')

if __name__ == "__main__":
    train_custom_model()