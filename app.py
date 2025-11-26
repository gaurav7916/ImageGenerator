import torch
from PIL import Image
import io
import streamlit as st
import base64
import os
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
import numpy as np
from diffusers import DiffusionPipeline
import warnings

warnings.filterwarnings("ignore")


class TextToImageEvaluator:
    def __init__(self):
        with st.spinner("Loading CLIP model..."):
            clip_model_name = "openai/clip-vit-large-patch14-336"
            tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_name)
            image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.clip_processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        with st.spinner("Loading image generation model..."):
            self.generator = DiffusionPipeline.from_pretrained(
                "Lykon/dreamshaper-xl-v2-turbo"
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        self.generator.to(self.device)

        if self.device == "cuda":
            self.generator.enable_attention_slicing()

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


@st.cache_resource
def get_evaluator():
    """Load the evaluator (cached to avoid reloading)"""
    return TextToImageEvaluator()


def main():
    st.set_page_config(
        page_title="Text-to-Image Generator",
        page_icon="🎨",
        layout="centered"
    )
    
    st.title("🎨 Text-to-Image Generator")
    st.markdown("Generate images from text prompts and see how well they match!")
    
    # Show device info
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"Running on: **{device}**")
    
    # Load model (cached)
    evaluator = get_evaluator()
    st.sidebar.success("Models loaded successfully!")
    
    # Input prompt
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="A beautiful sunset over mountains with a lake in the foreground",
        height=100
    )
    
    # Advanced settings in sidebar
    st.sidebar.header("⚙️ Settings")
    num_steps = st.sidebar.slider("Inference Steps", 10, 50, 30)
    guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 15.0, 7.5)
    
    # Generate button
    if st.button("🚀 Generate Image", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating image... This may take a moment."):
                try:
                    # Generate image
                    generated_image = evaluator.generate_image(
                        prompt, 
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale
                    )
                    
                    # Calculate scores
                    clip_score = evaluator.calculate_clip_score(generated_image, prompt)
                    geneval_score = clip_score * 2.5
                    
                    # Display results
                    st.image(generated_image, caption="Generated Image", use_container_width=True)
                    
                    # Display scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("CLIP Score", f"{clip_score:.4f}")
                    with col2:
                        st.metric("GenEval Score", f"{geneval_score:.4f}")
                    
                    # Download button
                    buffered = io.BytesIO()
                    generated_image.save(buffered, format="PNG")
                    st.download_button(
                        label="📥 Download Image",
                        data=buffered.getvalue(),
                        file_name="generated_image.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")


if __name__ == "__main__":
    main()
