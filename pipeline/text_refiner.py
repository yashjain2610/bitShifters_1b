# pipeline/text_refiner.py
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

class TextRefiner:
    """
    Refines a block of text using an abstractive summarization model (T5-small).
    """
    def __init__(self, models_dir):
        """
        Initializes the TextRefiner and loads the T5 model.
        Args:
            models_dir (str): Directory where models are saved.
        """
        print("Initializing TextRefiner...")
        model_path = os.path.join(models_dir, 't5-small')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"T5 model not found at {model_path}. Please run the download_models.py script.")

        # Use CPU, as per constraints
        self.device = 'cpu'
        print(f"Loading T5 model to {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        print("T5 model loaded successfully.")

    def refine(self, text, max_length=700, min_length=300):
        """
        Generates a refined, summary-like version of the input text.
        Args:
            text (str): The text to refine.
            max_length (int): The maximum length of the generated summary.
            min_length (int): The minimum length of the generated summary.

        Returns:
            str: The refined (summarized) text.
        """
        # Prepare the text for T5 by adding the summarization prefix
        prompt = "summarize: " + text
        
        # Tokenize the input
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        # Generate the summary
        summary_ids = self.model.generate(
            inputs, 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        
        # Decode the generated summary
        refined_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return refined_text