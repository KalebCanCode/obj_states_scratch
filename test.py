from pathlib import Path
import clip
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
import itertools
import IPython
from queries import q, a
from dicts import cat_dicts, video_list
import numpy as np
from pytube import YouTube
from pytube.cli import on_progress

# from IPython.display import display, Image

# data_dir = Path("uncut_vids")
device = torch.device("mps")
model_clip, preprocess = clip.load("ViT-L/14", device=device) 

text = "an apple"
s1_vector = model_clip.encode_text(clip.tokenize(text).to(device))
print(s1_vector)