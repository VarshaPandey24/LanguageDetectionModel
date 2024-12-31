import gradio as gr
from LanguageDetection import predict  # Import the predict function from your .py file

import os

# Install scikit-learn if not already installed
try:
    import sklearn
except ImportError:
    os.system("pip install sklearn")


# Define the prediction function wrapper for Gradio
def detect_language(text):
    if not text.strip():
        return "Please enter some text for language detection."
    
    try:
        # Use the predict function from the .py file
        predicted_language = predict(text)
        return f"Detected Language: {predicted_language}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=detect_language,
    inputs=gr.Textbox(label="Enter text for language detection", lines=2, placeholder="Type your text here..."),
    outputs=gr.Textbox(label="Detected Language"),
    title="Language Detection Model",
    description="A machine learning model that predicts the language of the given text.",
    examples=["Hello world", "Bonjour le monde", "Hola mundo"]
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
