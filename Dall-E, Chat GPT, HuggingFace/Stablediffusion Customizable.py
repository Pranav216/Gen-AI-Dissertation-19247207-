from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin, extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans"
# a still life art

image = pipe((prompt),
             generator =torch.Generator(device ='cuda').manual_seed(850), 
             num_inference_steps =999).images[0]

image.save("saved.png")
print("CustomResult Seed_150 Steps_10.png'")

