import base64
from pdf2image import convert_from_path
import openai
import pytesseract

# Set OpenAI API key
openai.api_key = "your_openai_api_key"

# Step 1: Convert PDF Pages to Base64 Encoded Images
def pdf_to_base64(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path)
    
    # Convert each page to base64
    base64_images = []
    for page in pages:
        # Save each page as a temporary image file
        page.save("temp_page.jpg", "JPEG")
        with open("temp_page.jpg", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            base64_images.append(base64_image)
    return base64_images

# Step 2: Analyze Base64 Encoded Images Using GPT-4 Vision
def analyze_image(base64_image, query):
    prompt = f"""
    The following is a base64-encoded image. Extract the relevant information based on this query: "{query}". Return a concise result.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-vision",  # Replace with the vision-enabled GPT-4 model if applicable
        messages=[
            {"role": "system", "content": "You are an assistant that processes images to extract relevant information."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": base64_image}
        ]
    )
    return response['choices'][0]['message']['content']

# Step 3: Relevance Scoring Function
def score_relevance(value, query):
    # Example: Score based on keyword matches
    keywords = query.lower().split()
    score = sum(1 for keyword in keywords if keyword in value.lower())
    return score

# Step 4: Process All Pages and Determine the Most Relevant Result
def process_pdf_with_relevance(pdf_path, query):
    # Convert PDF to base64-encoded images
    base64_images = pdf_to_base64(pdf_path)
    
    # Extract information from each page
    extracted_values = {}
    for i, base64_image in enumerate(base64_images):
        page_number = f"Page {i + 1}"
        extracted_values[page_number] = analyze_image(base64_image, query)
    
    # Score relevance for each extracted value
    scored_values = {page: score_relevance(value, query) for page, value in extracted_values.items()}
    
    # Determine the most relevant value
    most_relevant_page = max(scored_values, key=scored_values.get)
    most_relevant_value = extracted_values[most_relevant_page]
    
    return {
        "most_relevant_page": most_relevant_page,
        "most_relevant_value": most_relevant_value,
        "all_extracted_values": extracted_values,
        "relevance_scores": scored_values
    }

# Step 5: Run the Process
if __name__ == "__main__":
    pdf_path = "your_pdf_file.pdf"
    query = "Extract the most relevant date mentioned in the document."
    
    results = process_pdf_with_relevance(pdf_path, query)
    
    print(f"Most Relevant Page: {results['most_relevant_page']}")
    print(f"Most Relevant Value: {results['most_relevant_value']}")
    print("\nAll Extracted Values:")
    for page, value in results["all_extracted_values"].items():
        print(f"{page}: {value}")
    print("\nRelevance Scores:")
    for page, score in results["relevance_scores"].items():
        print(f"{page}: {score}")
