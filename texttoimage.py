# -*- coding: utf-8 -*-
!pip install --upgrade diffusers transformers -q

from pathlib import Path
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12
    auth_token = "your_huggingface_token"  # Update with your Hugging Face token

# Load the Stable Diffusion model
def load_image_gen_model():
    model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id,
        torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32,
        revision="fp16" if CFG.device == "cuda" else None,
        use_auth_token=CFG.auth_token
    )
    model = model.to(CFG.device)
    return model

# Function to generate images based on a text prompt
def generate_image(prompt, model):
    result = model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    )
    image = result.images[0]
    image = image.resize(CFG.image_gen_size)
    return image

# Initialize and generate an image
if __name__ == "__main__":
    image_gen_model = load_image_gen_model()
    prompt = "man under the apple tree"
    generated_image = generate_image(prompt, image_gen_model)
    
    # Display the generated image
    plt.figure(figsize=(6, 6))
    plt.imshow(generated_image)
    plt.axis("off")
    plt.show()
