
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_gif
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def genrate_image(query,image_name):
    pipe = DiffusionPipeline.from_pretrained("openai/shap-e")
    prompt = query
    image = pipe(
        prompt,
        frame_size=256 , # or 512
        num_inference_steps=50,
        guidance_scale=7.5

    ).images[0]
    gif_path = export_to_gif(image,image_name)
    image_np=np.array(image)
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()
    return gif_path


query = input("Enter the prompt for image generation : ")   
image_name = f"{query.replace(' ', '_')}.gif"
gif_path = genrate_image(query,image_name)
print(f"GIF saved to {gif_path}")

#re=genrate_image("dinasour","dinasour_3d.gif")
#print(f"GIF saved to {re}")

#in this project we are using diffusers library to generate 3d images from text prompt using shap-e model.


