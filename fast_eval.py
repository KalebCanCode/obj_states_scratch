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

class AgeRestrictedError(Exception):
    pass

def process_video(code, category, i):
    print(code)
    n_vid_path = 'www.youtube.com/watch?v=' + code
    yt=YouTube(n_vid_path,on_progress_callback=on_progress)
    try:
        videos=yt.streams.filter(file_extension = "mp4").first()
    except AgeRestrictedError:
        return None
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
    # cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # ret, frame = cap.read()
        if not ret:
            break
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


def constrained_argmax(pred_action, pred_state_init, pred_state_term, ret_best_worst=False):
    max_val_, best_idx_ = -1, (0, 0, 0)
    min_val, worst_idx = float('inf'), (0, 0, 0)
    for i in range(len(pred_state_init)):
        for j in range(i + 2, len(pred_state_term)):
            val_ = pred_state_init[i] * pred_state_term[j]
            k = torch.argmax(pred_action[i + 1:j]).item()
            val_ *= pred_action[i + 1 + k]
            if val_ > max_val_:
                best_idx_ = i, i + 1 + k, j
                max_val_ = val_
            
            if val_ < min_val:
                worst_idx = i, i + 1 + k, j
                min_val = val_
    
    if ret_best_worst:
        return best_idx_, max_val_, worst_idx, min_val
    else:
        return best_idx_, max_val_

def contrained_get_indices(image_vectors, s, a, ret_best_worst=False):
    s1_vector = model_clip.encode_text(clip.tokenize([s[0]]).to(device))
    a_vector = model_clip.encode_text(clip.tokenize([a]).to(device))
    s2_vector = model_clip.encode_text(clip.tokenize([s[1]]).to(device))

    sim_s1 = torch.cosine_similarity(image_vectors, s1_vector)
    sim_a = torch.cosine_similarity(image_vectors, a_vector)
    sim_s2 = torch.cosine_similarity(image_vectors, s2_vector)

    if ret_best_worst:
        return constrained_argmax(sim_a, sim_s1, sim_s2, True)
    else:
        return constrained_argmax(sim_a, sim_s1, sim_s2)


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
    print(indices)
    
    if indices[2] in d:
        if d[indices[2]]== 3:
            state_score += 0.5
    
    return action_score, state_score

class BreakLoopException(Exception):
    pass

def visualize_indices(video_name, category):
    state_descriptions = q[category]["states"]
    action_descriptions = q[category]["action"]
    w = 0
    z = process_video(video_name, category, w)
    image_vec, frame_count = z

    y_max = 0
    t_inices = ()
    for des in a["tile"]:
        indices, y = contrained_get_indices(image_vec, s=state_descriptions[des[0]], a=action_descriptions[des[1]], ret_best_worst=True)
        # z_list.append(z)
        # indices_list.append(indices)
        if y > y_max:
            y_max = y
            t_inices = indices


    return None


def evaluation():
        
    # for categories in cat_dicts:
        action_vec = []
        state_vec = []
        action_correct = 0
        state_correct =  0
        state_descriptions = q["tea"]["states"]
        action_descriptions = q["tea"]["action"]
        w = 0
        a_total = 0
        s_total = 0
        for videos in cat_dicts["tea"]:
            a_total += 1
            s_total += 1
            video_name = videos[0]
            video_annotation = videos[1]
            try:
                z = process_video(video_name, "tea", w)

                if z == None:
                    continue
                else:
                    image_vec, frame_count = z
            except BreakLoopException:
                # break_flag = True
                continue
            # increasing_sets = increasing_sets(frame_count)
            z_max = 0
            true_inices = ()
            for des in a["tea"]:
                indices, z = contrained_get_indices(image_vec, s=state_descriptions[des[0]], a=action_descriptions[des[1]])
                # z_list.append(z)
                # indices_list.append(indices)
                if z > z_max:
                    z_max = z
                    true_inices = indices
                    

            
            
            action_score, state_score = eval_one_clip(video_annotation, true_inices)
            action_correct += action_score
            state_correct += state_score
            print("video " + str(w) + " action:" + str(action_score))
            print("video " + str(w) + " state:" + str(state_score))
            action_vec.append(action_score)
            state_vec.append(state_score)
            w += 1
        
        category_action_precision = action_correct/ a_total
        # action_vec.append(category_action_precision)
        category_state_precision = state_correct/ s_total
        # state_vec.append(category_state_precision)

        with open('results.txt', 'w') as f:
            l = 0
            for ac, ss in zip(action_vec, state_vec):
                f.write("video " + str(l) + " action:" + str(ac) + "\n")
                f.write("video " + str(l) + " state:" + str(ss) + "\n")
                l +=1
                

            f.write("tea state:" + str(category_state_precision))
            f.write('\n')
            f.write("tea action:" + str(category_action_precision))
            f.write('\n')



            

if __name__ == "__main__":
    evaluation()















        

        








