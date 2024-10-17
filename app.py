from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import torch
import PyPDF2
import os
import logging
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Hugging Face API token - Ensure this is set in your environment
hf_token = os.getenv('ChatIIITA')
if hf_token:
    login(token=hf_token)
else:
    logger.error("Hugging Face API token is missing.")
    raise EnvironmentError("Hugging Face API token not set in environment.")

# Load LLaMA model and tokenizer with authentication
try:
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load model configuration
    config = LlamaConfig.from_pretrained(model_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, use_auth_token=hf_token, torch_dtype=torch.float16, device_map="auto")
except Exception as e:
    logger.error(f"Failed to initialize LLaMA model: {e}")
    raise

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise

# Load and process the PDF file
pdf_path = "data.pdf"  # Ensure this is the correct path to your PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Endpoint to generate embeddings
@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400
    try:
        inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
        return jsonify({"embedding": embeddings})
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": "Error generating embeddings"}), 500

# RAG-based chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Text field is required'}), 400
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(data['text'], return_tensors="pt").to(device)

        # Use the PDF text as context for the model
        context = pdf_text
        input_with_context = f"{context}\n\nUser: {data['text']}\nAI:"
        
        # Generate a response based on the PDF context
        output = model.generate(tokenizer.encode(input_with_context, return_tensors="pt").to(device), max_new_tokens=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True).split("AI:")[-1].strip()  # Get the AI response only
        
        return jsonify({"answer": response})
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
