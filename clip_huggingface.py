from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("open_beer_bottle.png")

inputs = processor(text=["bottle with a cap", "bottle with no cap"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

image.show()
print(probs)
