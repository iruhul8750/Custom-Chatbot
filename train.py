import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import logging
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers.optimization import AdamW
from nltk_utils import tokenize, stem, bag_of_words, preprocess_text
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.x_data = torch.FloatTensor(X_data)
        self.y_data = torch.LongTensor(y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


class ELECTRADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def expand_questions(patterns):
    """Generate variations of questions for better training"""
    expanded = []
    question_words = ['what', 'where', 'when', 'how', 'can you', 'could you', 'would you']

    for pattern in patterns:
        expanded.append(pattern)

        # Add variations with different question starters
        for qword in question_words:
            if not pattern.lower().startswith(qword):
                expanded.append(f"{qword.capitalize()} {pattern.lower()}")

        # Add variations with "tell me"
        if not pattern.lower().startswith('tell me'):
            expanded.append(f"Tell me {pattern.lower()}")

        # Add variations with "I want to know"
        expanded.append(f"I want to know {pattern.lower()}")

    return list(set(expanded))  # Remove duplicates


def prepare_training_data(intents_path='intents.json'):
    with open(intents_path) as f:
        intents = json.load(f)

    # Prepare data for Neural Network
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)

        # Expand patterns with variations
        expanded_patterns = expand_questions(intent['patterns'])

        for pattern in expanded_patterns:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    # Prepare data for ELECTRA
    texts = []
    electra_labels = []
    tag2idx = {tag: idx for idx, tag in enumerate(tags)}

    for intent in intents['intents']:
        expanded_patterns = expand_questions(intent['patterns'])
        for pattern in expanded_patterns:
            texts.append(pattern)
            electra_labels.append(tag2idx[intent['tag']])

    return (np.array(X_train), np.array(y_train), all_words, tags), (texts, electra_labels)


def train_neural_net(X_train, y_train, all_words, tags, save_path='data.pth', **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = {
        'num_epochs': 1500,  # Increased epochs
        'batch_size': 16,  # Increased batch size
        'learning_rate': 0.001,
        'hidden_size': 16,  # Increased hidden size
        **kwargs
    }

    input_size = len(X_train[0])
    output_size = len(tags)
    model = NeuralNet(input_size, params['hidden_size'], output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=True)

    model.train()
    logger.info(f"Training NeuralNet on {device}...")
    for epoch in range(params['num_epochs']):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            logger.info(f'Epoch [{epoch + 1}/{params["num_epochs"]}], Loss: {loss.item():.4f}')

    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": params['hidden_size'],
        "all_words": all_words,
        "tags": tags
    }, save_path)

    logger.info(f"\nNeuralNet training complete. Model saved to {save_path}")


def train_electra(texts, labels, save_path='electra_model.pth', **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = {
        'num_epochs': 5,  # Increased epochs
        'batch_size': 16,  # Increased batch size
        'learning_rate': 5e-5,
        **kwargs
    }

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraForSequenceClassification.from_pretrained(
        'google/electra-small-discriminator',
        num_labels=len(set(labels)))
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = ELECTRADataset(train_encodings, train_labels)
    val_dataset = ELECTRADataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    optimizer = AdamW(model.parameters(), lr=params['learning_rate'])

    logger.info(f"Training ELECTRA on {device}...")
    for epoch in range(params['num_epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        for batch in val_loader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        logger.info(f'Epoch [{epoch + 1}/{params["num_epochs"]}] - '
                    f'Train Loss: {total_loss / len(train_loader):.4f}, '
                    f'Val Loss: {val_loss / len(val_loader):.4f}')

    torch.save(model.state_dict(), save_path)
    logger.info(f"\nELECTRA training complete. Model saved to {save_path}")


if __name__ == "__main__":
    try:
        # Prepare data for both models
        nn_data, electra_data = prepare_training_data()
        X_train, y_train, all_words, tags = nn_data
        texts, electra_labels = electra_data

        # Train Neural Network
        train_neural_net(
            X_train, y_train, all_words, tags,
            save_path='data.pth',
            num_epochs=1500,
            batch_size=16,
            hidden_size=16,
            learning_rate=0.001
        )

        # Train ELECTRA
        train_electra(
            texts, electra_labels,
            save_path='electra_model.pth',
            num_epochs=5,
            batch_size=16,
            learning_rate=5e-5
        )

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise