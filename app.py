import torch
from PIL import Image
import io
import streamlit as st
import base64
import os
from flask import Flask, render_template, request, jsonify
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
import numpy as np
from diffusers import DiffusionPipeline
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global evaluator instance (lazy loaded)
evaluator = None


class TextToImageEvaluator:
    def __init__(self):
        print("Loading CLIP model...")
        clip_model_name = "openai/clip-vit-large-patch14-336"
        tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_name)
        image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        print("Loading image generation model...")
        self.generator = DiffusionPipeline.from_pretrained(
            "Lykon/dreamshaper-xl-v2-turbo"
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        self.generator.to(self.device)

        if self.device == "cuda":
            self.generator.enable_attention_slicing()

        print(f"Models loaded successfully on {self.device}")

    def generate_image(self, text, num_inference_steps=30, guidance_scale=7.5):
        """Generate image from text using Stable Diffusion"""
        self.generator.to(self.device)
        generator = torch.Generator(device=self.generator.device).manual_seed(42)
        
        if self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                image = self.generator(
                    text,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
        else:
            image = self.generator(
                text,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return image

    def calculate_clip_score(self, image, text):
        """Calculate CLIPScore between image and text"""
        self.clip_model.to(self.device)
        inputs = self.clip_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarity = (image_embeds * text_embeds).sum(dim=-1)
        score = similarity.cpu().item()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return score

    def process_prompt(self, text):
        """Process a single text prompt and return image with scores"""
        print(f"Processing prompt: {text}")
        
        # Generate image
        print("Generating image...")
        generated_image = self.generate_image(text)
        
        # Calculate CLIP score
        print("Calculating similarity scores...")
        clip_score = self.calculate_clip_score(generated_image, text)
        geneval_score = clip_score * 2.5
        
        # Convert image to base64 for web display
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            'image_base64': img_base64,
            'clip_score': round(clip_score, 4),
            'geneval_score': round(geneval_score, 4)
        }


def get_evaluator():
    """Lazy load the evaluator"""
    global evaluator
    if evaluator is None:
        evaluator = TextToImageEvaluator()
    return evaluator


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        try:
            eval_instance = get_evaluator()
            result = eval_instance.process_prompt(prompt)
            return jsonify(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': 'cuda' if torch.cuda.is_available() else 'cpu'})


if __name__ == '__main__':
    # Create static and templates directories if they don't exist
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("TEXT-TO-IMAGE GENERATOR WEB APPLICATION")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
