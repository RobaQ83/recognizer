import io
import os

from PIL import Image, ImageOps
from torchvision import transforms


def get_pixels_from(canvas):
    filename = ".tmp.png"

    ps = canvas.postscript(colormode="mono", colormap="colorMap")
    image = Image.open(io.BytesIO(ps.encode("utf-8")))

    ImageOps.invert(image).save(filename)

    width, height = 28, 28
    pixels = Image.open(filename).resize((width, height))

    os.remove(filename)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(pixels)
    return input_tensor.unsqueeze(0)
