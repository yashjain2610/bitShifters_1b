# download_models.py
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

def download_summarizer_model():
    """
    Downloads and saves the T5-small model for summarization.
    This script needs to be run once in an environment with internet access.
    """
    # --- Configuration ---
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    model_name = 't5-small'
    model_path = os.path.join(models_dir, model_name)

    # --- Download and Save T5 Summarizer ---
    if os.path.exists(model_path):
        print(f"Model '{model_name}' already exists at '{model_path}'. Skipping download.")
        return

    print(f"Downloading Summarizer: {model_name}...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"Summarizer saved successfully to '{model_path}'.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
    
if __name__ == "__main__":
    download_summarizer_model()