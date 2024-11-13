import fitz  # PyMuPDF for PDF extraction
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins, especially localhost for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict it to only your frontend origin, e.g., ['http://localhost:5173']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to get an answer based on the user's question and the PDF content
def get_answer_from_pdf(pdf_path, question):
    # Extract content from the PDF
    pdf_content = extract_text_from_pdf(pdf_path)
    
    # Prepare the context for the Groq model (PDF content + user question)
    context = f"Here is some content from the PDF:\n{pdf_content}\n\nThe user asks: {question}\nPlease answer based on the provided content."
    
    # Your Groq client
    client = Groq()

    # Call the Groq API to get a response
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Use the appropriate model
        messages=[{"role": "user", "content": context}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    # Output the generated result
    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""
    
    return answer.strip()

# Define a request model using Pydantic for validation
class QuestionRequest(BaseModel):
    question: str

# FastAPI route to receive the question and return the response
@app.post("/chat")
async def ask_question(request: QuestionRequest):
    try:
        # Path to the PDF (you can also make this dynamic if needed)
        pdf_path = '/Users/devruhela/SE_project/pythonBackend/Data01.pdf'

        # Get the answer from the model based on the PDF and the question
        answer = get_answer_from_pdf(pdf_path, request.question)
        
        # Return the answer as JSON response
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the app (run this command in terminal: `uvicorn app_name:app --reload`)
