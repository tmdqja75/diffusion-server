from io import BytesIO
from typing import Union, Optional # 타입정의 
from fastapi import FileResponse
from fastapi import FastAPI
from torch import autocast

from uuid import uuid4

app = FastAPI()
from PIL import Image

from diffusers import StableDiffusionPipeline




@app.get('/draw')
def drawing_pictures(prompt: Optional[str] = None): # prompt 자동 처라
    if torch.cuda.is_available():
        with autocast('cuda'):
            image = app.pipe(prompt).images[0]
    else:
        image = app.pipe(prompt).images[0]

    image_path = f'out/{uuid4()}.png'
    image.save(image_path)

    image = Image.open("image.jpg")

    # Save the image to a file-like object in memory
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Create a response object using the `FileResponse` class
    response = FileResponse(image_bytes, media_type="image/jpeg")

    return response
    

    


    

@app.on_event('startup')
def load_pipeline():
    # app.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # if torch.cuda.is_availabe():
    #     app.pipe.to('cuda')
    pass


@app.on_event('shutdown')
def addf():
    app.pipe = None
    pass


# server_address.com/draw?prompt=a-dog-flying-above-the-sea