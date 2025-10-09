from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64
import time
from openaillm import TextToImageEvaluator
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Initialize the image generator once at startup
print("Loading image generation model...")
image_generator = TextToImageEvaluator()
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def generate_image():
    try:
        prompt = request.form.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'Please enter a prompt'}), 400
        
        print(f"Generating image for prompt: {prompt}")
        
        # Generate image
        generated_image = image_generator.generate_image(
            prompt, 
            num_inference_steps=20,  # Reduced for faster generation
            guidance_scale=7.5
        )
        
        # Calculate CLIP score
        clip_score = image_generator.calculate_clip_score(generated_image, prompt)
        
        # Convert PIL image to base64 for web display
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response_text = f"✅ Image generated successfully!\nCLIP Score: {clip_score:.4f}\nPrompt: {prompt}"
        
        return jsonify({
            'response': response_text,
            'image_data': f"data:image/png;base64,{img_str}",
            'clip_score': f"{clip_score:.4f}"
        })
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({
            'error': f'Sorry, there was an error generating the image: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)