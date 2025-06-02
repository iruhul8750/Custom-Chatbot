import torch
import numpy as np
import random
import json
import logging
import os
import re
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch import nn
from nltk_utils import tokenize, stem, bag_of_words, preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        try:
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            logger.info("NeuralNet initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NeuralNet: {str(e)}")
            raise

    def forward(self, x):
        try:
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            return out
        except Exception as e:
            logger.error(f"Error in NeuralNet forward pass: {str(e)}")
            raise


class ELECTRAClassifier:
    def __init__(self, intents):
        try:
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            self.model = ElectraForSequenceClassification.from_pretrained(
                'google/electra-small-discriminator',
                num_labels=len(set(intent['tag'] for intent in intents['intents'])))
            self.tags = sorted(set(intent['tag'] for intent in intents['intents']))

            if os.path.exists('electra_model.pth'):
                try:
                    self.model.load_state_dict(torch.load('electra_model.pth'))
                    logger.info("Loaded fine-tuned ELECTRA weights")
                except Exception as e:
                    logger.error(f"Failed to load ELECTRA weights: {str(e)}")
                    raise

            logger.info("ELECTRA model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ELECTRAClassifier: {str(e)}")
            raise

    def predict(self, text):
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            return self.tags[pred.item()], confidence.item()
        except Exception as e:
            logger.error(f"Error in ELECTRA prediction: {str(e)}")
            return "unknown", 0.0


class NeuralNetWrapper:
    def __init__(self, model_path='data.pth'):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")

            data = torch.load(model_path)
            self.model = NeuralNet(
                input_size=data['input_size'],
                hidden_size=data['hidden_size'],
                num_classes=data['output_size']
            )
            self.model.load_state_dict(data['model_state'])
            self.model.eval()
            self.all_words = data['all_words']
            self.tags = data['tags']

            with open('intents.json') as f:
                self.intents = json.load(f)
            logger.info("NeuralNetWrapper initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NeuralNetWrapper: {str(e)}")
            raise

    def predict(self, sentence):
        try:
            if not sentence or not isinstance(sentence, str):
                return None, 0.0

            tokenized_sentence = tokenize(sentence)
            bow = bag_of_words(tokenized_sentence, self.all_words)
            bow = torch.from_numpy(bow).unsqueeze(0).float()

            with torch.no_grad():
                output = self.model(bow)

            probs = torch.softmax(output, dim=1)
            conf, predicted = torch.max(probs, dim=1)
            predicted_idx = predicted.item()

            if predicted_idx < len(self.tags):
                tag = self.tags[predicted_idx]
                for intent in self.intents['intents']:
                    if intent['tag'] == tag:
                        return random.choice(intent['responses']), conf.item()

            return None, 0.0
        except Exception as e:
            logger.error(f"Error in NeuralNet prediction: {str(e)}")
            return None, 0.0


class HybridChatModel:
    def __init__(self, intents_path='intents.json'):
        try:
            if not os.path.exists(intents_path):
                raise FileNotFoundError(f"Intents file {intents_path} not found")

            with open(intents_path) as f:
                self.intents = json.load(f)
                logger.info(f"Loaded intents from {intents_path}")

            # Initialize models
            self.models_initialized = {
                'neural_net': False,
                'electra': False,
                'fallback': True  # Fallback is always available
            }

            try:
                self.neural_net = NeuralNetWrapper()
                self.models_initialized['neural_net'] = True
            except Exception as e:
                logger.error(f"Failed to initialize NeuralNet: {str(e)}")
                self.neural_net = None

            try:
                self.electra = ELECTRAClassifier(self.intents)
                self.models_initialized['electra'] = True
            except Exception as e:
                logger.error(f"Failed to initialize ELECTRA: {str(e)}")
                self.electra = None

            self._init_fallback()
            logger.info(f"Model initialization status: {self.models_initialized}")

        except Exception as e:
            logger.error(f"Error initializing HybridChatModel: {str(e)}")
            raise

    def _init_fallback(self):
        """Enhanced fallback system with keyword matching"""
        self.location_keywords = ['location', 'address', 'where is', 'how to reach',
                                  'directions', 'find', 'located', 'place', 'venue']

        # Preprocess all patterns for faster matching
        self.processed_patterns = []
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                processed = preprocess_text(pattern)
                self.processed_patterns.append({
                    'processed': processed,
                    'response': random.choice(intent['responses']),
                    'tag': intent['tag']
                })

    def _get_fallback_response(self, text):
        """Enhanced fallback with location detection and pattern similarity"""
        text_lower = preprocess_text(text)

        # Check for location-related questions first
        if any(keyword in text_lower for keyword in self.location_keywords):
            for intent in self.intents['intents']:
                if intent['tag'] == 'school_location':
                    return random.choice(intent['responses'])

        # Find most similar preprocessed pattern
        best_match = None
        best_score = 0

        for pattern in self.processed_patterns:
            # Simple word overlap scoring
            score = len(set(text_lower.split()) & set(pattern['processed'].split()))
            if score > best_score:
                best_score = score
                best_match = pattern

        if best_match and best_score >= 2:  # At least 2 words in common
            return best_match['response']

        # Default fallback
        return random.choice([
            "I'm not sure I understand. Could you rephrase your question about Bagiya School?",
            "That's an interesting question! Could you ask about our location, admission, or curriculum?",
            "I'm still learning. Could you ask about our school hours, location, or admission process?"
        ])

    def get_initialization_status(self):
        return self.models_initialized

    def predict(self, text):
        try:
            if not text or not isinstance(text, str):
                return {
                    'response': "Please provide a valid message.",
                    'model': 'error',
                    'confidence': 0,
                    'status': 'invalid_input'
                }

            # Try NeuralNet first
            if self.models_initialized['neural_net']:
                nn_response, nn_conf = self.neural_net.predict(text)
                if nn_response and nn_conf > 0.65:  # Lowered confidence threshold
                    return {
                        'response': nn_response,
                        'model': 'neural_net',
                        'confidence': float(nn_conf),
                        'status': 'success'
                    }

            # Try ELECTRA if NeuralNet didn't return confident answer
            if self.models_initialized['electra']:
                electra_intent, electra_conf = self.electra.predict(text)
                if electra_conf > 0.75:  # Adjusted threshold
                    for intent in self.intents['intents']:
                        if intent['tag'] == electra_intent:
                            return {
                                'response': random.choice(intent['responses']),
                                'model': 'electra',
                                'confidence': float(electra_conf),
                                'status': 'success'
                            }

            # Fallback system
            fallback_response = self._get_fallback_response(text)
            return {
                'response': fallback_response,
                'model': 'fallback',
                'confidence': None,
                'status': 'fallback'
            }

        except Exception as e:
            logger.error(f"Error in hybrid prediction: {str(e)}")
            return {
                'response': "Sorry, I encountered an error processing your request.",
                'model': 'error',
                'confidence': 0,
                'status': 'prediction_error'
            }