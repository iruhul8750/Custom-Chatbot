import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import logging
from collections import defaultdict
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize stemmer
stemmer = PorterStemmer()

# Enhanced synonyms dictionary with more comprehensive coverage
SYNONYMS = {
    'location': ['address', 'place', 'whereabouts', 'site', 'venue', 'position', 'where is'],
    'find': ['locate', 'reach', 'directions to', 'navigate to', 'how to get to'],
    'school': ['institution', 'academy', 'college', 'campus', 'educational institute'],
    'fees': ['fee structure', 'tuition fees', 'extra fees', 'payment', 'cost', 'charges'],
    'admission': ['enrollment', 'registration', 'entry', 'application'],
    'curriculum': ['syllabus', 'course content', 'academic program', 'study plan']
}


def download_nltk_data():
    """Download required NLTK data with enhanced error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt data already available")
        return True
    except LookupError:
        logger.info("Downloading NLTK punkt data...")
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt data downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {str(e)}")
            logger.error(traceback.format_exc())
            return False


# Download required NLTK data at module import with verification
if not download_nltk_data():
    raise RuntimeError("Failed to initialize NLTK data - essential for tokenization")


def expand_with_synonyms(word):
    """Expand a word with its synonyms with improved matching"""
    if not isinstance(word, str) or not word.strip():
        return [word] if word else []

    word_lower = word.lower()
    for key, syns in SYNONYMS.items():
        if word_lower == key or word_lower in syns:
            return list(set([key] + syns))  # Remove duplicates
    return [word]


def tokenize(sentence):
    """Split sentence into array of words/tokens with enhanced error handling"""
    if not isinstance(sentence, str) or not sentence.strip():
        logger.warning("Empty or invalid sentence received for tokenization")
        return []

    try:
        tokens = nltk.word_tokenize(sentence.lower())
        expanded_tokens = []
        for token in tokens:
            if token.isalpha():  # Only process alphabetic tokens
                expanded_tokens.extend(expand_with_synonyms(token))
        return expanded_tokens
    except Exception as e:
        logger.error(f"Tokenization failed for sentence: '{sentence}'")
        logger.error(traceback.format_exc())
        return sentence.lower().split()  # Fallback to simple whitespace splitting


def stem(word):
    """Find the root form of the word with better validation"""
    if not isinstance(word, str) or not word.strip():
        return word if word else ""

    try:
        return stemmer.stem(word.lower())
    except Exception as e:
        logger.error(f"Stemming failed for word '{word}': {str(e)}")
        logger.error(traceback.format_exc())
        return word.lower()  # Fallback to lowercase


def bag_of_words(tokenized_sentence, all_words):
    """
    Enhanced bag of words with input validation and better error handling
    """
    if not isinstance(tokenized_sentence, list) or not isinstance(all_words, list):
        logger.error("Invalid input types for bag_of_words")
        return np.zeros(len(all_words), dtype=np.float32) if isinstance(all_words, list) else np.array([],
                                                                                                       dtype=np.float32)

    try:
        # Stem each word in the tokenized sentence with validation
        tokenized_sentence = [stem(w) for w in tokenized_sentence if isinstance(w, str)]

        # Convert all_words to lowercase and stem for consistent matching
        processed_all_words = [stem(w) for w in all_words if isinstance(w, str)]

        # Initialize bag with 0 for each word
        bag = np.zeros(len(processed_all_words), dtype=np.float32)

        # Create a frequency dictionary for faster lookup
        token_freq = defaultdict(int)
        for word in tokenized_sentence:
            token_freq[word] += 1

        # Set to 1 for each word that exists in the sentence
        for idx, w in enumerate(processed_all_words):
            if token_freq.get(w, 0) > 0:
                bag[idx] = 1.0

        return bag
    except Exception as e:
        logger.error(f"Bag of words generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return np.zeros(len(all_words), dtype=np.float32) if isinstance(all_words, list) else np.array([],
                                                                                                       dtype=np.float32)


def preprocess_text(text):
    """Enhanced text preprocessing with input validation"""
    if not isinstance(text, str):
        logger.warning(f"Non-string input received for preprocessing: {type(text)}")
        return ""

    try:
        tokens = tokenize(text)
        return ' '.join(tokens) if tokens else ""
    except Exception as e:
        logger.error(f"Text preprocessing failed for input: '{text}'")
        logger.error(traceback.format_exc())
        return text.lower() if isinstance(text, str) else ""


if __name__ == "__main__":
    # Enhanced test cases
    test_cases = [
        "Where is the school located?",
        "What are the admission fees?",
        "",
        12345,
        None,
        "How to reach the campus?"
    ]

    words = ["location", "school", "find", "address", "fees", "admission", "campus"]

    for sentence in test_cases:
        print("\n" + "=" * 50)
        print(f"Testing with: {sentence} (Type: {type(sentence)})")

        try:
            print("Original:", sentence)
            tokenized = tokenize(sentence)
            print("Tokenized with synonyms:", tokenized)

            stemmed = [stem(w) for w in tokenized] if tokenized else []
            print("Stemmed:", stemmed)

            bag = bag_of_words(tokenized, words)
            print("Bag of words:", bag)
        except Exception as e:
            print(f"Error processing: {str(e)}")