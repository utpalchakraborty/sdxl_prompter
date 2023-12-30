from PIL import Image

from gradio_prompter.esrgan_model import UpscalerESRGAN

original = Image.open("test.png")
print(original.size)

upscaler = UpscalerESRGAN()

upscaled_image = upscaler.upscale(
    original, 4, "H:\\ai\\stable-diffusion-webui\\models\\ESRGAN\\4x-UltraSharp.pth"
)
upscaled_image.save("test_upscaled.png")
