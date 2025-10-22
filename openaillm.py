import torch
from datasets import load_dataset
import time
import matplotlib.pyplot as plt
from PIL import Image
import tqdm as notebook_tqdm
from tqdm import tqdm
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
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
import base64
from io import BytesIO


# 1. Time model loading
# ---------------------
start = time.time()
# Load model directly
model = "openai/clip-vit-large-patch14-336"
tokenizer = CLIPTokenizerFast.from_pretrained(model)
image_processor = CLIPImageProcessor.from_pretrained(model)
model = CLIPModel.from_pretrained(model)
processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
# ------------------------
# 2. Time dataset loading
# ------------------------
dataset = load_dataset("NinaKarine/t2i-compbench", "3d_spatial_val")
end = time.time()
print(f"Dataset loaded in {end - start:.2f} seconds")


api_key ='AIzaSyALHqGrE8hRFm9_KC6VR5Q2VUr8iFkTa3I'

def setup_gemini(api_key: Optional[str] = None) -> genai:
    api_key ='AIzaSyALHqGrE8hRFm9_KC6VR5Q2VUr8iFkTa3I'
    genai.configure(api_key=api_key)
    return genai

def enhance_prompt_with_gemini(prompt: str, model_name: str = "models/gemini-2.5-pro") -> str:
    """
    Enhance a prompt using Gemini 2.5 Pro

    Args:
        prompt (str): The original prompt to enhance
        model_name (str): Gemini model to use

    Returns:
        str: Enhanced prompt
    """
    try:
        setup_gemini()

        # Initialize the model
        model = genai.GenerativeModel(model_name)

        # Enhanced prompt within 100 words limit
        enhancement_prompt = f"""
        Enhance this image generation prompt to be more detailed and real. Add specifics about visual style, composition, lighting, colors.
        Focus on key artistic elements that would improve image generation to look real. Keep the enhanced version under 50 words.

        Original: "{prompt}"

        Return ONLY the enhanced prompt, no explanations.
        Enhanced prompt:
        """
        response = model.generate_content(enhancement_prompt)
        enhanced = response.text.strip()

        # Verify word count
        if len(enhanced.split()) > 50:
            words = enhanced.split()[:50]
            enhanced = ' '.join(words)

        return enhanced

    except Exception as e:
        print(f"Error enhancing prompt with Gemini: {e}")
        return prompt


class TextToImageEvaluator:
    def __init__(self):
        # Load CLIP model for similarity evaluation
        self.clip_model = model
        self.clip_processor = processor

        # Load image generation model (Stable Diffusion)
        self.generator = DiffusionPipeline.from_pretrained(
            "Lykon/dreamshaper-xl-v2-turbo"
        )
        # Move to GPU if available and set appropriate dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move models to device
        self.clip_model.to(self.device)
        self.generator.to(self.device)

        # Enable memory efficient attention if on CUDA
        if self.device == "cuda":
            self.generator.enable_attention_slicing()
            # self.generator.enable_memory_efficient_attention()

        # Initialize inception model as None (lazy loading)
        self._inception_model = None


    def cleanup_memory(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            # Move models to CPU to free VRAM
            if hasattr(self, 'generator'):
                self.generator.to("cpu")
            if hasattr(self, 'clip_model'):
                self.clip_model.to("cpu")
            if hasattr(self, '_inception_model') and self._inception_model is not None:
                self._inception_model.to("cpu")
            
            torch.cuda.empty_cache()

    def generate_image(self, text, num_inference_steps=30, guidance_scale=7.5):
        """Generate image from text using Stable Diffusion"""
        # Use autocast for mixed precision on GPU
        generator = torch.Generator(device=self.generator.device).manual_seed(42)
        if self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                image = self.generator(
                    text,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator  # for reproducibility
                ).images[0]
        else:
            image = self.generator(
                text,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
    
        # Free VRAM
        if self.device == "cuda":
            self.generator.to("cpu")
            torch.cuda.empty_cache()

        return image

    def calculate_clip_score(self, image, text):
        """Calculate CLIPScore between image and text"""
        # Preprocess inputs
        self.clip_model.to(self.device)
        inputs = self.clip_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        # Calculate cosine similarity
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity (CLIPScore)
        similarity = (image_embeds * text_embeds).sum(dim=-1)
        score = similarity.cpu().item()
        self.clip_model.to("cpu")  # free VRAM
        torch.cuda.empty_cache()
        return score

    def evaluate_similarity(self, generated_image, original_text):
        """Evaluate similarity using multiple metrics"""
        results = {}

        # CLIPScore with original text
        clip_score = self.calculate_clip_score(generated_image, original_text)
        results['clip_score'] = clip_score

        # GenEval metric (CLIPScore normalized by text embedding norm)
        geneval_score = clip_score * 2.5
        results['geneval_score'] = geneval_score
        return results

    def process_dataset(self, dataset, num_samples=10, save_images=False):
        """Process dataset and generate evaluation metrics"""
        scores = {
            'clip_scores': [],
            'geneval_scores': [],
            'generated_images': [],
            'texts': []
        }
        for i, example in enumerate(dataset['spatial_val']):
            if i >= num_samples:
                break
            print("=" * 50)
            print("GEMINI 2.5 PRO PROMPT ENHANCEMENT")
            print("=" * 50)
            print(f"--- Iteration {i+1} ---")
            enhanced_text = enhance_prompt_with_gemini(example['text'])
            print(f"Original prompt:  {example['text']}")
            print(f"Enhanced prompt:  {enhanced_text}")
            print(f"Word count: {len(enhanced_text.split())}")
            print(f"Processing sample {i+1}/{num_samples}: {enhanced_text}")
            try:
                # Generate image
                generated_image = self.generate_image(enhanced_text)
                # Calculate similarity scores
                similarity_results = self.evaluate_similarity(generated_image, example['text'])
                # Store results
                scores['clip_scores'].append(similarity_results['clip_score'])
                scores['geneval_scores'].append(similarity_results['geneval_score'])
                scores['generated_images'].append(generated_image)
                scores['texts'].append(enhanced_text)

                # Save image if requested
                if save_images:
                    safe_filename = f"generated_image_{i}.png"
                    generated_image.save(safe_filename)
                    print(f"  Saved as: {safe_filename}")

                print(f"  CLIPScore: {similarity_results['clip_score']:.4f}")
                print(f"  GenEval: {similarity_results['geneval_score']:.4f}")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return scores

    def calculate_average_similarity(self, scores):
        """Calculate average similarity metrics"""
        if not scores['clip_scores']:
            return {
                'average_clip_score': 0.0,
                'average_geneval_score': 0.0,
                'num_samples': 0
            }
        avg_clip = np.mean(scores['clip_scores'])
        avg_geneval = np.mean(scores['geneval_scores'])
        return {
            'average_clip_score': avg_clip,
            'average_geneval_score': avg_geneval,
            'num_samples': len(scores['clip_scores'])
        }

    def visualize_results(self, scores, num_examples=5):
        """Visualize generated images with their scores"""
        if not scores['generated_images']:
            print("No images to visualize")
            return
        num_examples = min(num_examples, len(scores['generated_images']))

        # Create figure with subplots
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3 * num_examples))
        if num_examples == 1:
            axes = [axes]

        for i in range(num_examples):
            # Show generated image
            axes[i][0].imshow(scores['generated_images'][i])
            axes[i][0].set_title(f"Image {i+1}\n'{scores['texts'][i][:30]}...'", fontsize=10)
            axes[i][0].axis('off')
            # Show scores
            bars = axes[i][1].bar(['CLIP', 'GenEval'], [scores['clip_scores'][i], scores['geneval_scores'][i]], color=['skyblue', 'lightcoral'])
            axes[i][1].set_ylabel('Score')
            axes[i][1].set_title('Similarity Scores', fontsize=10)
            axes[i][1].set_ylim(0, max(scores['clip_scores'][i], scores['geneval_scores'][i]) * 1.2)

            # Add value labels on bars
            for bar, v in zip(bars, [scores['clip_scores'][i], scores['geneval_scores'][i]]):
                axes[i][1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _prepare_inception_model(self):
        """Lazy-load inception v3 for Inception Score calculations."""
        if self._inception_model is not None:
            return self._inception_model
        # Load pretrained inception v3
        model = inception_v3(pretrained=True, transform_input=False)
        model.eval()
        model.to(self.device)
        # Remove the final fully connected layer to get features
        model.fc = torch.nn.Identity()
        self._inception_model = model
        return model

    def _images_to_tensor_batch(self, pil_images, size=299):
        """Convert list of PIL images to a tensor batch suitable for inception."""
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        tensors = [transform(img).unsqueeze(0) for img in pil_images]
        return torch.cat(tensors, dim=0).to(self.device)

    def calculate_inception_score(self, pil_images, batch_size=8, splits=10):
        """
        Compute Inception Score for a list of PIL images.
        Returns (mean_IS, std_IS) across splits.
        """
        if len(pil_images) == 0:
            return (0.0, 0.0)
        model = self._prepare_inception_model()
        # Get class probabilities for each image
        preds = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i+batch_size]
                x = self._images_to_tensor_batch(batch, size=299)
                # Get Inception v3 predictions
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = model(x)
                else:
                    features = model(x)
                # Apply softmax to get probabilities
                prob = F.softmax(features, dim=1)
                preds.append(prob.cpu())
        preds = torch.cat(preds, dim=0).numpy()  # shape (N, 1000)

        # Compute IS across splits
        N = preds.shape[0]
        split_scores = []
        split_size = max(1, floor(N / splits))  # Ensure at least 1 sample per split
        for k in range(splits):
            start_idx = k * split_size
            end_idx = min((k + 1) * split_size, N)
            if start_idx >= end_idx:
                break
            part = preds[start_idx:end_idx]
            py = part.mean(axis=0, keepdims=True)  # marginal
            # KL divergence for each image
            kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
            sum_kl = kl.sum(axis=1)
            split_score = np.exp(np.mean(sum_kl))
            split_scores.append(split_score)

        if not split_scores:
            return (0.0, 0.0)

        mean_is = float(np.mean(split_scores))
        std_is = float(np.std(split_scores))
        return mean_is, std_is

    def calculate_entropy_of_inception(self, pil_images, batch_size=8):
        """
        Calculate average entropy of the Inception predicted distribution for images.
        Higher entropy -> predictions are spread (less confident).
        """
        if len(pil_images) == 0:
            return 0.0
        model = self._prepare_inception_model()
        entropies = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i+batch_size]
                x = self._images_to_tensor_batch(batch, size=299)
                logits = model(x)
                prob = F.softmax(logits, dim=1)
                ent = - (prob * torch.log(prob + 1e-12)).sum(dim=1)
                entropies.append(ent.cpu().numpy())
        entropies = np.concatenate(entropies, axis=0)
        return float(np.mean(entropies))

    def calculate_diversity_score(self, pil_images, batch_size=8):
        """
        Calculate diversity using CLIP image embeddings:
        mean pairwise cosine distance between generated images.
        Returns average pairwise distance in [0, 2] (cosine distance).
        """
        if len(pil_images) < 2:
            return 0.0

        # Use existing CLIP encoder already present in the class
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i+batch_size]
                inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True).to(self.device)
                out = self.clip_model.get_image_features(**{k: v for k, v in inputs.items() if k in ['pixel_values'] or True})
                # Many CLIP wrappers differ; assume image_embeds attribute or get_image_features
                if hasattr(out, "image_embeds"):
                    embeds = out.image_embeds
                else:
                    # if we used get_image_features, out is tensor
                    embeds = out if isinstance(out, torch.Tensor) else out[0]
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeds.append(embeds.cpu())
        all_embeds = torch.cat(all_embeds, dim=0)  # (N, D)

        # pairwise cosine distances
        # cos_sim matrix -> distance = 1 - cos_sim (in [0,2] if not normalized, but for normalized it's [0,2]? For normalized, cos in [-1,1] so distance in [0,2])
        sim_matrix = all_embeds @ all_embeds.T  # (N,N)
        n = sim_matrix.shape[0]
        # take upper triangle without diagonal
        idxs = torch.triu_indices(n, n, offset=1)
        sims = sim_matrix[idxs[0], idxs[1]].numpy()
        cos_distances = 1.0 - sims  # in [0,2]
        mean_pairwise_distance = float(np.mean(cos_distances))
        return mean_pairwise_distance

    def aggregate_benchmarks(self, scores):
        """
        Compute a collection of benchmark metrics given the 'scores' dict returned by process_dataset.
        This includes:
         - average CLIP and GenEval (already present)
         - Inception Score (mean & std) computed on generated images
         - Diversity score (pairwise CLIP distance)
         - Entropy of Inception predictions
        Returns a dict with these aggregated values.
        """
        results = {}
        agg_sim = self.calculate_average_similarity(scores)
        results.update(agg_sim)
        pil_images = scores.get('generated_images', [])
        if pil_images:
            try:
                mean_is, std_is = self.calculate_inception_score(pil_images, batch_size=8, splits= min(10, max(1, len(pil_images))))
                results['inception_score_mean'] = mean_is
                results['inception_score_std'] = std_is
            except Exception as e:
                results['inception_score_mean'] = None
                results['inception_score_std'] = None
                print("Inception score failed:", e)

            try:
                results['inception_entropy'] = self.calculate_entropy_of_inception(pil_images)
            except Exception as e:
                results['inception_entropy'] = None
                print("Inception entropy failed:", e)

            try:
                results['diversity_score'] = self.calculate_diversity_score(pil_images)
            except Exception as e:
                results['diversity_score'] = None
                print("Diversity score failed:", e)
        else:
            results['inception_score_mean'] = 0.0
            results['inception_score_std'] = 0.0
            results['inception_entropy'] = 0.0
            results['diversity_score'] = 0.0

        return results

    def show_image_grid_with_benchmarks(self, scores, ncols=4, filename='image_grid.png'):
        """
        Create a grid of generated images annotated with CLIP, GenEval,
        and (if available) inference of Inception predicted label top-1.
        """
        imgs = scores.get('generated_images', [])
        if not imgs:
            print("No images to show.")
            return
        n = len(imgs)
        ncols = min(ncols, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            if i >= n:
                ax.axis('off')
                continue
            img = imgs[i]
            ax.imshow(img)
            clip_s = scores['clip_scores'][i] if i < len(scores['clip_scores']) else None
            gene_s = scores['geneval_scores'][i] if i < len(scores['geneval_scores']) else None
            text = scores['texts'][i] if i < len(scores.get('texts', [])) else ''
            caption = f"CLIP: {clip_s:.3f}\nGenEval: {gene_s:.3f}\n'{text[:60]}...'"
            ax.set_title(caption, fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        


def main():
    print(f"Dataset loaded: {len(dataset['spatial_val'])} samples")
    # Initialize evaluator
    print("\nInitializing Text-to-Image Evaluator...")
    evaluator = TextToImageEvaluator()
    print(f"Using device: {evaluator.device}")
    # Process dataset
    scores = evaluator.process_dataset(
        dataset,
        num_samples=1,  # Use a number divisible by 2 for nice grid layouts
        save_images=True
    )
    # Calculate average similarity and comprehensive benchmarks
    print("\n" + "="*60)
    print("CALCULATING COMPREHENSIVE BENCHMARKS")
    print("="*60)
    avg_scores = evaluator.calculate_average_similarity(scores)
    benchmark_results = evaluator.aggregate_benchmarks(scores)

    # Generate comprehensive visualizations
    if avg_scores['num_samples'] > 0:
        try:
            # 4. Image grid with benchmarks
            print("\n4. Generated Images Grid...")
            evaluator.show_image_grid_with_benchmarks(
                scores,
                ncols=min(3, avg_scores['num_samples']),
                filename='comprehensive_image_grid.png'
            )
        except Exception as e:
            print(f"Image grid failed: {e}")

        # Print detailed scores
        print("\n" + "="*60)
        print("DETAILED SAMPLE SCORES")
        print("="*60)
        for i, (enhanced_text, clip_score, geneval_score) in enumerate(zip(
            scores['texts'], scores['clip_scores'], scores['geneval_scores'])):
            print(f"{i+1:2d}. CLIP: {clip_score:.4f}, GenEval: {geneval_score:.4f}")
            print(f"    Text: {enhanced_text[:90]}{'...' if len(enhanced_text) > 90 else ''}")

    else:
        print("No samples were successfully processed.")

    # Display results
    print("\n" + "="*60)
    print("BENCHMARK METRICS")
    print("="*60)
    print(f"Number of samples processed: {avg_scores['num_samples']}")
    print(f"Average CLIPScore: {avg_scores['average_clip_score']:.4f}")
    print(f"Average GenEval Score: {avg_scores['average_geneval_score']:.4f}")

    # Display benchmark results if available
    if benchmark_results['inception_score_mean'] is not None:
        print(f"Inception Score: {benchmark_results['inception_score_mean']:.4f} ± {benchmark_results['inception_score_std']:.4f}")
    else:
        print("Inception Score: Failed to compute")

    if benchmark_results['inception_entropy'] is not None:
        print(f"Inception Entropy: {benchmark_results['inception_entropy']:.4f}")
    else:
        print("Inception Entropy: Failed to compute")

    if benchmark_results['diversity_score'] is not None:
        print(f"Diversity Score: {benchmark_results['diversity_score']:.4f}")
    else:
        print("Diversity Score: Failed to compute")


if __name__ == "__main__":
    main()
