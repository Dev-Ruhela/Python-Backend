from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import requests
import os
import logging
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Hugging Face API token - Ensure this is set in your environment
hf_token = os.getenv('ChatIIITA')
if hf_token is None:
    logger.error("Hugging Face API token is missing.")
    raise EnvironmentError("Hugging Face API token not set in environment.")

# Hugging Face model API URL
model_api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B"

# Function to make requests to Hugging Face API
def call_hugging_face_api(input_text):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": input_text}
    
    response = requests.post(model_api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()  # Return the response as is; adjust according to your API response structure
    else:
        logger.error(f"API call failed: {response.status_code} - {response.text}")
        return {"error": "Error calling Hugging Face API"}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""  # Handle None case
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise

# Load and process the PDF file
pdf_path = "data.pdf"  # Ensure this is the correct path to your PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Endpoint to generate embeddings (now calls Hugging Face API)
@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400
    try:
        # Call the Hugging Face API for embeddings
        response = call_hugging_face_api(data['text'])
        if 'embedding' in response:
            return jsonify({"embedding": response['embedding']})
        else:
            return jsonify(response), 500
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": "Error generating embeddings"}), 500

# RAG-based chat endpoint (now calls Hugging Face API)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400
    try:
        # Use the PDF text as context for the model
        context = pdf_text
        input_with_context = f"{context}\n\nUser: {data['text']}\nAI:"
        
        # Call the Hugging Face API for chat
        response = call_hugging_face_api(input_with_context)
        if 'generated_text' in response:
            return jsonify({"answer": response['generated_text']})
        else:
            return jsonify(response), 500
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({"error": "Error processing chat request"}), 500

# Root endpoint
@app.route('/')
def root():
    return jsonify({"message": "LLaMA Model Service is Running"})

# Endpoint to fetch PDF data
@app.route('/pdf_data')
def get_pdf_data():
    return jsonify({"pdf_text": pdf_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
