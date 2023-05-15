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
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device) 

# video_path = data_dir / 'Art In Apple Flower _ Fruit Carving Garnish _ Apple Art _ Party Garnishing-4__D0XdFT9Q.mp4'
# video_path = data_dir / "apple_test.mp4"
video_path = Path("eval_videos")

def process_video(code, category, i):
    n_vid_path = 'www.youtube.com/watch?v=' + code
    yt=YouTube(n_vid_path,on_progress_callback=on_progress)
    videos=yt.streams.filter(file_extension = "mp4").first()
    filename = category + str(i) + ".mp4"
    videos.download(output_path=video_path, filename=filename)
    p = video_path / filename

    cap = cv2.VideoCapture(str(p))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    seconds = round(frame_count / fps)
    frame_count = seconds
    image_vectors = torch.zeros((frame_count + 1, 512), device=device)
    n = 0
    i = 0
    frame_array = []
    while True:
        ret, frame = cap.read()
        if n%fps == 0:
            frame_array.append(frame)
            with torch.no_grad():
                image_vectors[i] = model_clip.encode_image(
                    preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                )
                i+=1
        n+=1
        if ret == False:
            break;

    cap.release()
    cv2.destroyAllWindows()

    return image_vectors, frame_count


#For now use a deterministic list for apple cutting video 
states_and_action_apple = ["an apple", "cutting of an apple", "pieces of an apple"]
shoe_label = ["a photo of a shoe that is not tied", "a photo of tying a shoe", "a photo of a finish tied shoe"]
avocad_label = ["half watermelon", "cutting a watermelon", "sliced watermelon"]


def increasing_sets(frames):
    # Generate all possible combinations of size 3 of increasing indices from 0 to 24
    index_combinations = itertools.combinations(range(frames), 3)
    increasing_sets = []

    # Iterate through each combination
    for combination in index_combinations:
        # Check if the combination is increasing
        if combination[0] < combination[1] < combination[2]:
            increasing_sets.append(combination)

    return increasing_sets


def get_frame_indicies(image_vectors, s, a, frames):
    indicies = ()
    label = 0
    orderings = increasing_sets(frames)
    s1_vector = model_clip.encode_text(clip.tokenize([s[0]]).to(device))
    a_vector = model_clip.encode_text(clip.tokenize([a]).to(device))
    s2_vector = model_clip.encode_text(clip.tokenize([s[1]]).to(device))

    sim_s1 = torch.cosine_similarity(image_vectors, s1_vector)
    sim_a = torch.cosine_similarity(image_vectors, a_vector)
    sim_s2 = torch.cosine_similarity(image_vectors, s2_vector)


    for o in orderings:
        i = o[0]
        j = o[1]
        k = o[2]

        z = sim_s1[i] * sim_a[j] * sim_s2[k]

        if z > label:
            label = z
            indicies = o
    
    return indicies, z

# def display_frame(index: list):
    
#     for i in index:
#         # cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         # ret, frame = cap.read()
#         m = Image.fromarray(frame_array[i])
#         m.show()





# indices = get_frame_indicies(states_and_action_apple  , frame_count)
# print(indices)

# display_frame(indices)

def eval_one_clip(d, indices):
    action_score = 0
    state_score = 0
    if d[indices[0]] == 1:
        state_score += 0.5
    if d[indices[1]] == 2:
        action_score += 1
    if d[indices[2]] == 3:
        state_score += 0.5
    return action_score, state_score


def evaluation():
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    #WRITE RESULTS TO TEXT FILE
    action_vec = []
    state_vec = []
    for categories in cat_dicts:
        action_correct = 0
        state_correct =  0
        state_descriptions = q[categories]["states"]
        action_descriptions = q[categories]["action"]


        w = 0
        a_total = 0
        s_total = 0
        for videos in categories:
            a_total += 1
            s_total += 1
            video_name = videos[0]
            video_annotation = videos[1]
            image_vec, frame_count = process_video(video_name, categories, w)
            increasing_sets = increasing_sets(frame_count)
            z_list =[]
            indices_list = []
            for des in a[categories]:
                indices, z = get_frame_indicies(image_vec, s=state_descriptions[des[0]], a=action_descriptions[des[1]], frames=frame_count)
                z_list.append(z)
                indices_list.append(indices)
            
            true_inices = indices_list[np.argmax(np.array(z_list))]
            action_score, state_score = eval_one_clip(video_annotation, true_inices)
            action_correct += action_score
            state_correct += state_score

        
        
        category_action_precision = action_correct/ a_total
        action_vec.append(category_action_precision)
        category_state_precision = state_correct/ s_total
        state_vec.append(category_state_precision)

        with open('results.txt', 'w') as f:
            f.write(categories + "state:" + str(category_state_precision))
            f.write('\n')
            f.write(categories + "action:" + str(category_action_precision))
            f.write('\n')

    
    total_action_precision = np.mean(np.array(action_vec))
    total_state_precision = np.mean(np.array(state_vec))

    with open('results.txt', 'w') as f:
            f.write(categories + "state:" + str(total_action_precision))
            f.write('\n')
            f.write(categories + "action:" + str(total_state_precision))

            

if __name__ == "__main__":
    evaluation()















        

        








