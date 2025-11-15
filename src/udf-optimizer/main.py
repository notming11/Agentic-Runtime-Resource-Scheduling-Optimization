import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Available models:
# models/gemma-3-12b-it
# models/gemma-3-27b-it
# models/gemma-3n-e4b-it
# models/gemma-3n-e2b-it
# models/gemini-flash-latest
# models/gemini-flash-lite-latest
# models/gemini-pro-latest
# models/gemini-2.5-flash-lite
# models/gemini-2.5-flash-image-preview
# models/gemini-2.5-flash-image
# models/gemini-2.5-flash-preview-09-2025
# models/gemini-2.5-flash-lite-preview-09-2025
# models/gemini-robotics-er-1.5-preview
# models/gemini-2.5-computer-use-preview-10-2025

genai.configure(api_key=api_key)

# Create the model
model = genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')

# Generate content
response = model.generate_content("Hello, world!")

print(response.text)
