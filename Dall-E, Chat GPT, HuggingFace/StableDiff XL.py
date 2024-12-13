from diffusers import DiffusionPipeline
import torch

# Load Stable Diffusion XL Base1.0
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         torch_dtype=torch.float16,variant="fp16", use_safetensors=True).to("cuda")

# Optional CPU offloading to save some GPU Memory
pipe.enable_model_cpu_offload()

# identifier = "London Eye wheel"
# prompt = f"{identifier} behind the Empire State building"

prompt= "dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin, extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans."

image = pipe(
    prompt = prompt,
    num_inference_steps=50,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

# image.save(f"{prompt}'1'.jpg")