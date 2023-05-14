# import required module
import sys
import os
import pandas as pd
# assign directory
directory = '/Users/kalebnew/Desktop/obj_states_scratch/ChangeIt-main/annotations'

video_list = []

 
# iterate over files in
# that directory
i = 0
c = 0
cat_dicts = {}
apple_list =  []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isdir(f):
        cat_dicts[i] = []
        for r in sorted(os.listdir(f)):
            df = pd.read_csv(os.path.join(directory, filename, r), header=None, index_col=0)
            rr = os.path.basename(os.path.join(directory, filename, r)).split('.')[0]
            video_list.append(rr)
            d = df.to_dict()
            e = (rr, d[1])
            cat_dicts[i].append(e)
            ff = os.path.join(directory, r)
    i+=1

# print(video_list)
print(cat_dicts[0][5])

    