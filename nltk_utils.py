import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize stemmer
stemmer = PorterStemmer()

# Synonyms dictionary for better pattern matching
SYNONYMS = {
    'location': ['address', 'place', 'whereabouts', 'site', 'venue', 'position'],
    'find': ['locate', 'reach', 'directions to', 'navigate to'],
    'school': ['institution', 'academy', 'college', 'campus'],
    'fees' : ['fees', 'fees structure', 'fees per year', 'tution fees', 'extra fees']
}


def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt data...")
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt data downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {str(e)}")
            raise


# Download required NLTK data at module import
download_nltk_data()


def expand_with_synonyms(word):
    """Expand a word with its synonyms"""
    for key, syns in SYNONYMS.items():
        if word == key or word in syns:
            return [key] + syns
    return [word]


def tokenize(sentence):
    """Split sentence into array of words/tokens with synonym expansion"""
    try:
        tokens = nltk.word_tokenize(sentence.lower())
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.extend(expand_with_synonyms(token))
        return expanded_tokens
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        return sentence.lower().split()  # Fallback to simple whitespace splitting


def stem(word):
    """Find the root form of the word"""
    try:
        return stemmer.stem(word.lower())
    except Exception as e:
        logger.error(f"Stemming failed for word '{word}': {str(e)}")
        return word.lower()  # Fallback to lowercase


def bag_of_words(tokenized_sentence, all_words):
    """
    Enhanced bag of words that accounts for synonyms
    """
    try:
        # Stem each word in the tokenized sentence
        tokenized_sentence = [stem(w) for w in tokenized_sentence]

        # Initialize bag with 0 for each word
        bag = np.zeros(len(all_words), dtype=np.float32)

        # Set to 1 for each word that exists in the sentence
        for idx, w in enumerate(all_words):
            if w in tokenized_sentence:
                bag[idx] = 1.0

        return bag
    except Exception as e:
        logger.error(f"Bag of words generation failed: {str(e)}")
        return np.zeros(len(all_words), dtype=np.float32)


def preprocess_text(text):
    """Enhanced text preprocessing with synonym expansion"""
    tokens = tokenize(text)
    return ' '.join(tokens)


if __name__ == "__main__":
    # Test the functions
    sentence = "Where is the school located?"
    words = ["location", "school", "find", "address"]

    print("Original:", sentence)
    tokenized = tokenize(sentence)
    print("Tokenized with synonyms:", tokenized)

    stemmed = [stem(w) for w in tokenized]
    print("Stemmed:", stemmed)

    bag = bag_of_words(tokenized, words)
    print("Bag of words:", bag)