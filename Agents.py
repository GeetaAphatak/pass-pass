from langchain.agents import initialize_agent, Tool
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms import OpenAI
from pdf2image import convert_from_path
import pytesseract

# Step 1: Define the PDF-to-Text Conversion Tool
def pdf_to_text(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path)
    
    # Extract text using Tesseract
    text_content = []
    for page in pages:
        text = pytesseract.image_to_string(page)
        text_content.append(text)
    
    return "\n".join(text_content)

# Step 2: Define Tools for the Agent
pdf_tool = Tool(
    name="PDFProcessor",
    func=lambda pdf_path: pdf_to_text(pdf_path),
    description="Extracts text from PDF files containing images."
)

# GPT-4 Tool
llm = OpenAI(model="gpt-4", temperature=0)  # Specify GPT-4

# Step 3: Initialize the Agent
tools = [pdf_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Step 4: Use the Agent
pdf_file_path = "example.pdf"
query = "Extract names, dates, and summaries from this PDF."

response = agent.run(f"Process this PDF: {pdf_file_path}. {query}")
print(response)
