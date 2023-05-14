from pathlib import Path

import clip
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm

sns.set_theme()
torch.set_printoptions(sci_mode=False)

data_dir = Path("data_dir")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device) 

image = preprocess(Image.open("images/cutting_apple_image_2.jpeg")).unsqueeze(0).to(device)

action_correct_text = clip.tokenize(["cutting an apple"]).to(device)
action_wrong1_text = clip.tokenize(["sliced apple"]).to(device)
action_wrong2_text = clip.tokenize(["skyscraper"]).to(device)

with torch.no_grad():
    image_vector = model_clip.encode_image(image)
    action_correct_vector = model_clip.encode_text(action_correct_text)
    aw1_vector = model_clip.encode_text(action_wrong1_text )
    aw2_vector = model_clip.encode_text(action_wrong2_text)

ac_similarity = torch.cosine_similarity(image_vector, action_correct_vector).item()
aw1_similarity = torch.cosine_similarity(image_vector, aw1_vector).item()
aw2_similarity = torch.cosine_similarity(image_vector, aw2_vector).item()

print(f"correct_action similarity: {ac_similarity:.2f}")
print(f"other_action similarity: {aw1_similarity:.2f}")
print(f"misc similarity: {aw2_similarity:.2f}")