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
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model_clip, preprocess = clip.load("ViT-L/14", device=device) 
# model_clip, preprocess = clip.load("ViT-B/32", device=device) 

# video_path = data_dir / 'Art In Apple Flower _ Fruit Carving Garnish _ Apple Art _ Party Garnishing-4__D0XdFT9Q.mp4'
# video_path = data_dir / "apple_test.mp4"
video_path = Path("eval_vids")

class AgeRestrictedError(Exception):
    pass

def process_video(code, category, i, ret_fa=False):
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
    image_vectors = torch.zeros((frame_count + 1, 768), device=device)
    # image_vectors = torch.zeros((frame_count + 1, 512), device=device)
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
            # print(frame)
            frame_array.append(frame)
            with torch.no_grad():
                image_vectors[i] = model_clip.encode_image(
                    preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                )
                # print(image_vectors[i])
                i+=1
        n+=1
        if ret == False:
            break

    cap.release()
    cv2.destroyAllWindows()

    if ret_fa:
        return image_vectors, frame_count, frame_array
    else:
        return image_vectors, frame_count



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
            # print(i, i + 1 + k, j, val_)
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
    # print(image_vectors)

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


def eval_one_clip(d, indices):
    print(indices)

    action_score = 0
    state_score = 0
    if d[indices[0]] == 1:
        state_score += 0.5
    if indices[1] in d:
        if d[indices[1]] == 2:
            action_score += 1
    
    
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
    z = process_video(video_name, category, w, True)
    image_vec, frame_count, frame_array = z
    y_max = 0
    p_min = float('inf')
    true_indices = ()
    worst_indices = ()
    for des in a["tile"]:
        b_indices, y, w_indices, p = contrained_get_indices(image_vec, s=state_descriptions[des[0]], a=action_descriptions[des[1]], ret_best_worst=True)
        # z_list.append(z)
        # indices_list.append(indices)
        if y > y_max:
            y_max = y
            true_indices = b_indices
        if p < p_min:
            p_min = p
            worst_indices = w_indices


    for i in true_indices:
        # cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # ret, frame = cap.read()
        m = Image.fromarray(frame_array[i])
        m.show()
    
    for i in worst_indices:
        # cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # ret, frame = cap.read()
        m = Image.fromarray(frame_array[i])
        m.show()



def evaluation(class_word, file):
        
    # for categories in cat_dicts:
        action_vec = []
        state_vec = []
        action_correct = 0
        state_correct =  0
        state_descriptions = q[class_word]["states"]
        action_descriptions = q[class_word]["action"]
        w = 0
        a_total = 0
        s_total = 0
        for videos in cat_dicts[class_word]:
            a_total += 1
            s_total += 1
            video_name = videos[0]
            video_annotation = videos[1]
            try:
                z = process_video(video_name, class_word, w)

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
            for des in a[class_word]:
                # print(des)
                indices, z = contrained_get_indices(image_vec, s=state_descriptions[des[0]], a=action_descriptions[des[1]])
                # print(indices, z)

                if z.item() > z_max:
                    print(z.item())
                    z_max = z.item()
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

        with open(file, 'w') as f:
            l = 0
            for ac, ss in zip(action_vec, state_vec):
                f.write("video " + str(l) + " action:" + str(ac) + "\n")
                f.write("video " + str(l) + " state:" + str(ss) + "\n")
                l +=1
                

            f.write(class_word +  " state:" + str(category_state_precision))
            f.write('\n')
            f.write(class_word + " action:" + str(category_action_precision))
            f.write('\n')



            

if __name__ == "__main__":
    x = [["outlet", "outlet_large_results.txt"]]
    # x = ["pan", "pan_large_results.txt"]
    for p in x:
        print(p)
        evaluation(p[0], p[1])
    # visualize_indices("-gOU6laeDog", "beer")
    # print(clip.available_models())
    # print(device)



