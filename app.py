from flask import Flask, render_template, request, jsonify
import torch
from model import HybridChatModel
import logging
import traceback
import time
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_required_files():
    required_files = ['intents.json']
    optional_files = ['data.pth', 'electra_model.pth']

    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        logger.error(f"Missing required files: {', '.join(missing)}")
        return False

    available_optional = [f for f in optional_files if os.path.exists(f)]
    logger.info(f"Available optional model files: {', '.join(available_optional) or 'None'}")

    return True


def initialize_chat_model():
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            if not check_required_files():
                raise RuntimeError("Missing required files for model initialization")

            model = HybridChatModel()
            status = model.get_initialization_status()
            logger.info(f"Model initialization status: {status}")

            if not any(status.values()):
                raise RuntimeError("No models were successfully initialized")

            return model
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise


try:
    logger.info("Initializing chat model...")
    chat_model = initialize_chat_model()
    logger.info("Chat model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chat model: {str(e)}")
    logger.error(traceback.format_exc())
    chat_model = None


@app.route("/")
def home():
    try:
        model_status = chat_model.get_initialization_status() if chat_model else None
        return render_template("index.html", model_status=model_status)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return "An error occurred while loading the chat interface.", 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request"}), 400

        user_text = data["message"].strip()
        if not user_text:
            return jsonify({"error": "Message cannot be empty"}), 400

        if chat_model is None:
            return jsonify({
                "answer": "The chat service is currently unavailable. Please try again later.",
                "model_used": "error",
                "confidence": 0,
                "status": "service_unavailable"
            }), 503

        response_data = chat_model.predict(user_text)
        return jsonify({
            "answer": response_data['response'],
            "model_used": response_data['model'],
            "confidence": response_data.get('confidence', 0),
            "status": response_data['status']
        })

    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "answer": "Sorry, I encountered an error processing your request.",
            "model_used": "error",
            "confidence": 0,
            "status": "server_error"
        }), 500


@app.route("/status")
def status():
    try:
        if chat_model is None:
            return jsonify({
                "status": "unavailable",
                "message": "Chat model not initialized",
                "initialized_models": []
            }), 503

        status = chat_model.get_initialization_status()
        return jsonify({
            "status": "available",
            "message": "Chat service is running",
            "initialized_models": [k for k, v in status.items() if v]
        })
    except Exception as e:
        logger.error(f"Error in status route: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.after_request
def after_request(response):
    try:
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    except Exception as e:
        logger.error(f"Error in after_request: {str(e)}")
        return response


if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        logger.error(traceback.format_exc())