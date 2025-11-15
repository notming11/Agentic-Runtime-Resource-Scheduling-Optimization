import os
import google.generativeai as genai
from dotenv import load_dotenv
import datetime
import json

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(script_dir, 'example_prompt.txt')
response_path = os.path.join(script_dir, 'example_response.txt')

# Read the prompt template from the file
try:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template_full = f.read()
except FileNotFoundError:
    print(f"Error: The file was not found at {prompt_path}")
    exit()


# Split the template into system and user parts
try:
    system_prompt_template, user_question = prompt_template_full.split("Research question:")
except ValueError:
    print("Error: 'Research question:' separator not found in the prompt file.")
    exit()

# Define variables
current_time = datetime.datetime.now().strftime("%c")
max_step_num = 10
locale = "en-US"

# Replace placeholders in the system prompt
system_prompt = system_prompt_template.replace("{{ CURRENT_TIME }}", current_time)
system_prompt = system_prompt.replace("{{ max_step_num }}", str(max_step_num))
system_prompt = system_prompt.replace("{{ locale }}", locale)


# Configure the model with the system prompt and to expect JSON output
generation_config = genai.types.GenerationConfig(
    response_mime_type="application/json"
)
model = genai.GenerativeModel(
    'models/gemini-2.5-flash-preview-09-2025',
    system_instruction=system_prompt,
    generation_config=generation_config
)


print("--- Sending Prompt to Gemini ---")

# Generate content with streaming, passing only the user question
response = model.generate_content(user_question.strip(), stream=True)

print("--- Gemini Response (also saving to example_response.txt) ---")

full_response_text = ""
try:
    with open(response_path, 'w', encoding='utf-8') as f:
        for chunk in response:
            # The response chunks are streamed as text
            chunk_text = chunk.text
            print(chunk_text, end='', flush=True)
            f.write(chunk_text)
            full_response_text += chunk_text
    print() # for a final newline

    # --- Post-processing the full response ---
    print("\n--- Parsed JSON Output ---")
    
    # Clean up potential markdown formatting if the model adds it
    if full_response_text.strip().startswith("```json"):
        cleaned_json_text = full_response_text.strip()[7:-3].strip()
    else:
        cleaned_json_text = full_response_text.strip()

    # Parse the complete JSON string
    parsed_json = json.loads(cleaned_json_text)
    
    # Pretty-print the parsed JSON object
    print(json.dumps(parsed_json, indent=2))

except Exception as e:
    print(f"\nAn error occurred during response handling: {e}")
    print("Full response text received:")
    print(full_response_text)
