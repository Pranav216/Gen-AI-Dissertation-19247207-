from diffusers import DiffusionPipeline
import torch

# Load Stable Diffusion XL Base1.0
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Optional CPU offloading to save some GPU Memory
pipe.enable_model_cpu_offload()

# Loading Trained DreamBooth LoRA Weights
# pipe.load_lora_weights("AdamLucek/sdxl-base-1.0-greenchair-dreambooth-lora")
pipe.load_lora_weights("BalcerK/sdxl-base-1.0-misa-dreambooth-lora")
identifier = "a photo of a candle"
prompt = f"{identifier} on fire."

# Invoke pipeline to generate image
image = pipe(
    prompt = prompt,
    num_inference_steps=50,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save(f"Sample_Custom.jpg")