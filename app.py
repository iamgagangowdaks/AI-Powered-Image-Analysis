from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GENAI_API_KEY")  # Ensure GENAI_API_KEY is set in .env
if not api_key:
    st.error("API key not found. Please set the GENAI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to get Gemini response for image analysis
def get_gemini_response(input_text, image):
    try:
        if not input_text.strip():
            return "Please provide a valid input prompt."
        
        # Convert PIL image to bytes
        img_bytes = image.convert("RGB")
        
        # Generate content using Gemini
        response = model.generate_content([input_text, img_bytes])
        
        if response and response.candidates:
            parts = response.candidates[0].content.parts
            return ' '.join(part.text for part in parts)
        else:
            return "No meaningful response generated."
    except Exception as e:
        return f"Error: {e}"

# Streamlit page configuration
st.set_page_config(page_title="Gemini Image Analysis", layout="wide")
st.header("Gemini Computer Vision Application")

# User Input
input_prompt = st.text_input("Input Prompt:", key="input", placeholder="Describe what you want to know about the image.")
uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image:", use_container_width=True)

    # Button to analyze the image
    if st.button("Analyze Image"):
        with st.spinner("Analyzing the image..."):
            response = get_gemini_response(input_prompt, image)
        
        # Display the response
        st.subheader("Analysis Result:")
        st.write(response)
