import os
# Suppress TensorFlow warnings BEFORE importing any TF-dependent libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['KMP_WARNINGS'] = '0'

# Additional TensorFlow suppression
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import io
import sys
stderr_backup = sys.stderr
sys.stderr = io.StringIO()
# Model loading code here
sys.stderr = stderr_backup


import torch
from datasets import load_dataset
import time
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
import numpy as np
from diffusers import DiffusionPipeline
import warnings
from torchvision.models import inception_v3
from torchvision import transforms as T
import torch.nn.functional as F
from math import floor
import google.generativeai as genai
from typing import Optional
import base64
from io import BytesIO

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS

warnings.filterwarnings("ignore")

# Restore stderr after imports that trigger TensorFlow
sys.stderr = stderr_backup

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Get Gemini API Key from environment variable for security
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBQGGvsxE-Ol3oMdAvNQDKiQJ0HpLyMm7I')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'AIzaSyBQGGvsxE-Ol3oMdAvNQDKiQJ0HpLyMm7I')


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Text to Image Generation Agentic AI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.2/socket.io.js"></script>
  
  <style>
    body {
      font-family: 'Manrope', sans-serif;
      margin: 0 auto 40px;
      background-color: #101626;
      color: #fff;
    }

    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }

    .loading {
        cursor: wait;
    }
    
    .loading * {
        pointer-events: none;
    }

    .generated-image {
      max-width: 100%; 
      height: auto;
      border-radius: 10px;
      margin: 20px 0;
    }

    #generate-image-container {
      display: flex;
      justify-content: center;
      margin: 20px 0;
    }

    #generate-image-btn, #enhance-prompt-btn {
      background-color: #140b9d;
      color: rgb(249, 247, 254);
      border: none;
      padding: 10px 20px;
      border-radius: 5px;
      font-size: 16px;
      cursor: pointer;
      transition: background-color 0.3s;
    }

    #generate-image-btn:hover, #enhance-prompt-btn:hover {
      background-color: #0e0770;
    }

    #generate-image-btn:disabled, #enhance-prompt-btn:disabled {
      background-color: #6c757d;
      cursor: not-allowed;
    }
  </style>
</head>

<body class="w-lg-50 w-md-75 w-sm-100 py-3">
  <main class="flex-shrink-0">
    <div>
      <h2 class="mt-3 p-3">Text to Image Generation Agentic AI</h2>
      <p class="p-3">Powered by Gemini 2.5 Pro Prompt Enhancement & Stable Diffusion
        <span style="font-weight: bold;">⏳ Image generation may take 30-60 seconds. Please be patient.</span>
      </p>

      <div class="input-group w-lg-50 w-md-75 w-sm-100 p-3" style="margin: 0 auto;">
        <input type="text" class="form-control" id="chat-input" placeholder="✨ Enter Your Prompt..."
          style="background-color: #3A4556; border: #3A4556; color: #fff;">
        <div class="input-group-append">
          <button id="enhance-prompt-btn" class="btn">Enhance Prompt</button>
        </div>
      </div>

      <div id="enhanced-prompt-container" class="p-3" style="display: none;">
        <div class="enhanced-prompt">
          <h6>Enhanced Prompt:</h6>
          <p id="enhanced-prompt-text"></p>
        </div>
      </div>

      <div id="status-message" style="display: none; margin: 10px; padding: 10px; border-radius: 5px;"></div>
      
      <div id="generate-image-container" style="display: none;">
        <button id="generate-image-btn">Generate Image</button>
      </div>
      
      <div id="image-placeholder" class="text-center my-4">
        <div id="no-image" style="
            width: 300px;
            height: 300px;
            background-color: #2e3545;
            border: 2px dashed #555;
            color: #aaa;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            border-radius: 10px;">
          <span>No image generated yet</span>
        </div>
        <div id="generated-image-container" style="display: none;">
          <img id="generated-image" class="generated-image" src="" alt="Generated Image">
          <div id="image-info" class="mt-2">
            <p id="clip-score" class="mb-1"></p>
            <p id="prompt-info" class="mb-1"></p>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
   // Dynamically get the server URL
    const serverUrl = window.location.origin;
    
    const socket = io(serverUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    const chatInput = document.getElementById('chat-input');
    const enhancePromptBtn = document.getElementById('enhance-prompt-btn');
    const generateImageBtn = document.getElementById('generate-image-btn');
    const enhancedPromptContainer = document.getElementById('enhanced-prompt-container');
    const enhancedPromptText = document.getElementById('enhanced-prompt-text');
    const generateImageContainer = document.getElementById('generate-image-container');
    const noImage = document.getElementById('no-image');
    const generatedImageContainer = document.getElementById('generated-image-container');
    const generatedImage = document.getElementById('generated-image');
    const clipScore = document.getElementById('clip-score');
    const promptInfo = document.getElementById('prompt-info');
    const statusMessage = document.getElementById('status-message');

    let currentEnhancedPrompt = '';

    function showStatus(message, type = 'info') {
      statusMessage.textContent = message;
      statusMessage.className = `status-message status-${type}`;
      statusMessage.style.display = 'block';
    }

    function hideStatus() {
      statusMessage.style.display = 'none';
    }

    function setLoading(isLoading) {
      if (isLoading) {
        document.body.classList.add('loading');
        enhancePromptBtn.disabled = true;
        generateImageBtn.disabled = true;
      } else {
        document.body.classList.remove('loading');
        enhancePromptBtn.disabled = false;
        generateImageBtn.disabled = false;
      }
    }

    enhancePromptBtn.addEventListener('click', () => {
      const prompt = chatInput.value.trim();
      
      if (!prompt) {
        showStatus('Please enter a prompt first', 'error');
        return;
      }

      showStatus('Enhancing prompt with Gemini...', 'info');
      setLoading(true);
      socket.emit('enhance_prompt', { prompt: prompt });
    });
    
    generateImageBtn.addEventListener('click', () => {
      if (!currentEnhancedPrompt) {
        showStatus('No enhanced prompt available', 'error');
        return;
      }

      showStatus('Generating image... This may take 30-60 seconds', 'info');
      setLoading(true);
      socket.emit('generate_image', { prompt: currentEnhancedPrompt });
    });

    socket.on('enhance_prompt_result', (data) => {
      setLoading(false);
      
      if (data.success) {
        currentEnhancedPrompt = data.enhanced_prompt;
        enhancedPromptText.textContent = currentEnhancedPrompt;
        enhancedPromptContainer.style.display = 'block';
        generateImageContainer.style.display = 'flex';
        showStatus('Prompt enhanced successfully!', 'success');
        setTimeout(hideStatus, 3000);
      } else {
        showStatus('Error enhancing prompt: ' + data.error, 'error');
      }
    });

    socket.on('generate_image_result', (data) => {
      setLoading(false);
      
      if (data.success) {
        noImage.style.display = 'none';
        generatedImage.src = data.image_data;
        generatedImage.alt = 'Generated image for: ' + data.prompt;
        generatedImageContainer.style.display = 'block';
        
        clipScore.textContent = 'CLIP Score: ' + (data.clip_score ? data.clip_score.toFixed(4) : 'N/A');
        promptInfo.textContent = 'Prompt: ' + data.prompt;
        
        showStatus('Image generated successfully!', 'success');
        setTimeout(hideStatus, 5000);
      } else {
        showStatus('Error generating image: ' + data.error, 'error');
      }
    });

    socket.on('generation_progress', (data) => {
      showStatus(data.message, 'info');
    });

    socket.on('connect', () => {
      console.log('Connected to server');
    });

    socket.on('disconnect', () => {
      showStatus('Disconnected from server', 'error');
    });
  </script>
</body>
</html>
'''

CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25
)

# Load CLIP model
print("Loading CLIP model...")
start = time.time()
model_name = "openai/clip-vit-large-patch14-336"
tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
end = time.time()
print(f"CLIP model loaded in {end - start:.2f} seconds")

# Load dataset (optional for web app)
print("Loading dataset...")
dataset = load_dataset("NinaKarine/t2i-compbench", "3d_spatial_val")
print(f"Dataset loaded: {len(dataset['spatial_val'])} samples")


def setup_gemini(api_key: Optional[str] = None):
    if api_key is None:
        api_key = GEMINI_API_KEY
    genai.configure(api_key=api_key)
    return genai


def enhance_prompt_with_gemini(prompt: str, model_name: str = "models/gemini-2.0-flash-exp") -> str:
    try:
        setup_gemini()
        model = genai.GenerativeModel(model_name)

        enhancement_prompt = f"""
        Enhance this image generation prompt to be more detailed and vivid. Add specifics about visual style, composition, lighting, colors, and mood.
        Focus on key artistic elements that would improve AI image generation. Keep the enhanced version under 50 words.

        Original: "{prompt}"

        Return ONLY the enhanced prompt, no explanations.
        Enhanced prompt:
        """

        response = model.generate_content(enhancement_prompt)
        enhanced = response.text.strip()

        if len(enhanced.split()) > 50:
            words = enhanced.split()[:50]
            enhanced = ' '.join(words)

        return enhanced

    except Exception as e:
        print(f"Error enhancing prompt with Gemini: {e}")
        return prompt


class TextToImageEvaluator:
    def __init__(self):
        self.clip_model = clip_model
        self.clip_processor = processor

        print("Loading Stable Diffusion model...")
        self.generator = DiffusionPipeline.from_pretrained(
            "Lykon/dreamshaper-xl-v2-turbo"
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        self.generator.to(self.device)

        if self.device == "cuda":
            self.generator.enable_attention_slicing()

        self._inception_model = None
        print(f"TextToImageEvaluator initialized on {self.device}")

    def cleanup_memory(self):
        if self.device == "cuda":
            if hasattr(self, 'generator'):
                self.generator.to("cpu")
            if hasattr(self, 'clip_model'):
                self.clip_model.to("cpu")
            if hasattr(self, '_inception_model') and self._inception_model is not None:
                self._inception_model.to("cpu")
            torch.cuda.empty_cache()

    def generate_image(self, text, num_inference_steps=30, guidance_scale=7.5):
        generator_torch = torch.Generator(device=self.generator.device).manual_seed(42)
        
        if self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                image = self.generator(
                    text,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator_torch
                ).images[0]
        else:
            image = self.generator(
                text,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator_torch
            ).images[0]
    
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return image

    def calculate_clip_score(self, image, text):
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


# Initialize evaluator
print("Initializing TextToImageEvaluator...")
evaluator = TextToImageEvaluator()


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@socketio.on('enhance_prompt')
def handle_enhance_prompt(data):
    try:
        print(f"Received enhance_prompt event with data: {data}")
        prompt = data['prompt']
        print(f"Enhancing prompt: {prompt}")
        
        enhanced_prompt = enhance_prompt_with_gemini(prompt)
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        emit('enhance_prompt_result', {
            'success': True,
            'enhanced_prompt': enhanced_prompt
        })
    except Exception as e:
        print(f"Error in handle_enhance_prompt: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('enhance_prompt_result', {
            'success': False,
            'error': str(e)
        })


@socketio.on('generate_image')
def handle_generate_image(data):
    try:
        prompt = data['prompt']
        print(f"Generating image for prompt: {prompt}")
        
        emit('generation_progress', {'message': 'Starting image generation...'})
        
        generated_image = evaluator.generate_image(prompt)
        
        emit('generation_progress', {'message': 'Calculating CLIP score...'})
        
        clip_score = evaluator.calculate_clip_score(generated_image, prompt)
        
        image_data = image_to_base64(generated_image)
        
        emit('generate_image_result', {
            'success': True,
            'image_data': image_data,
            'clip_score': clip_score,
            'prompt': prompt
        })
        
        print(f"Image generated successfully. CLIP Score: {clip_score:.4f}")
        
    except Exception as e:
        print(f"Error in handle_generate_image: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('generate_image_result', {
            'success': False,
            'error': str(e)
        })


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == "__main__":    
    port = int(os.environ.get('PORT', 5500))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("\n" + "="*60)
    print(f"Starting Flask-SocketIO server on {host}:{port}...")
    print("="*60 + "\n")
    
    # For production, set debug=False
    socketio.run(
        app, 
        debug=False,  # Changed to False for production
        port=port, 
        host=host,
        allow_unsafe_werkzeug=True  # Only if using Werkzeug in production (not recommended)
    )
