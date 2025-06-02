
# Bagiya School Chatbot

![Bagiya School Logo](static/images/school-logo.png)

A hybrid AI-powered chatbot for Bagiya School that assists with answering common queries about the school, admissions, fees, and more.

## Features

- **Hybrid AI Model**: Combines Neural Network and ELECTRA transformer for accurate responses
- **Quick Questions**: Predefined common questions for easy access
- **Emoji Support**: Built-in emoji picker for expressive communication
- **Responsive Design**: Works well on both desktop and mobile devices
- **Error Handling**: Robust error handling and fallback responses
- **Typing Indicators**: Visual feedback when the bot is processing

## Technologies Used

### Backend
- Python 3.x
- Flask (Web Framework)
- PyTorch (Deep Learning)
- Transformers (ELECTRA Model)
- NLTK (Natural Language Processing)

### Frontend
- HTML5, CSS3, JavaScript
- Font Awesome (Icons)
- Google Fonts (Nunito)

## Installation

### Prerequisites
- Python 3.7+
- pip
- virtualenv (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bagiya-school-chatbot.git
   cd bagiya-school-chatbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

5. **Train the models (optional)**
   ```bash
   python train.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the chatbot**
   Open `http://localhost:5000` in your browser

## Project Structure

```
bagiya-school-chatbot/
├── static/
│   ├── css/
│   │   └── style.css           # Chatbot styles
│   ├── js/
│   │   └── app.js              # Chatbot frontend logic
│   └── images/                 # Image assets
├── templates/
│   └── index.html              # Main HTML file
├── app.py                      # Flask application
├── model.py                    # Hybrid AI model implementation
├── nltk_utils.py               # NLP utility functions
├── train.py                    # Model training script
├── intents.json                # Training data and responses
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

The chatbot can be configured by modifying these files:

- `intents.json`: Add/update questions and responses
- `static/css/style.css`: Customize the look and feel
- `model.py`: Adjust model parameters and thresholds

## Usage

1. Click the chat icon in the bottom right corner to open the chat interface
2. Type your question or click one of the quick questions
3. The chatbot will respond with relevant information
4. Use the emoji picker to add emojis to your messages

## Training the Models

To retrain the models with updated data:

1. Update `intents.json` with new questions and responses
2. Run the training script:
   ```bash
   python train.py
   ```
3. The trained models will be saved as:
   - `data.pth` (Neural Network)
   - `electra_model.pth` (ELECTRA model)

## Troubleshooting

- **NLTK data not found**: Run `python -c "import nltk; nltk.download('punkt')"`
- **Model loading errors**: Check if `data.pth` and `electra_model.pth` exist
- **Port already in use**: Change the port in `app.py` (default: 5000)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact:
- Your Name - Ruhul Islam
- Project Link: https://github.com/iruhul8750/Custom-Chatbot.git
```

### Key Features of this README:
1. **Clear Project Description**: Immediately explains what the project does
2. **Visual Elements**: Includes space for your school logo
3. **Comprehensive Installation Guide**: Step-by-step setup instructions
4. **Project Structure**: Visual representation of the code organization
5. **Usage Instructions**: How to interact with the chatbot
6. **Training Information**: For updating the AI models
7. **Troubleshooting**: Common issues and solutions
8. **Contributing Guidelines**: For open-source collaboration
9. **Contact Information**: For support and questions

You can customize this template further by:
- Adding screenshots of your chatbot
- Including specific deployment instructions if hosted
- Adding more details about your school's specific requirements
- Updating the contact information with real details