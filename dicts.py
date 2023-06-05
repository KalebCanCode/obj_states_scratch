# import required module
import sys
import os
import pandas as pd
from queries import q, a
# assign directory
directory = 'ChangeIt-main/annotations'

video_list = []

 
# iterate over files in
# that directory
i = 0
c = 0
cat_dicts = {}
apple_list =  []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # print(f)
    
    if os.path.isdir(f):
        z =os.path.basename(f).split('/')[0]
        
        cat_dicts[z] = []
        for r in sorted(os.listdir(f)):
            df = pd.read_csv(os.path.join(directory, filename, r), header=None, index_col=0)
            rr = os.path.basename(os.path.join(directory, filename, r)).split('.')[0]
            video_list.append(rr)
            d = df.to_dict()
            e = (rr, d[1])
            cat_dicts[z].append(e)
            ff = os.path.join(directory, r)
    i+=1

# for categories in cat_dicts:
#         state_descriptions = q[categories]["states"]
#         action_descriptions = q[categories]["action"]
#         w = 0
#         print(state_descriptions)
#         print(action_descriptions)
#         for i in a[categories]:
#              print(i)
             
#              r = state_descriptions[i[0]] 
#              e = action_descriptions[i[1]]
#         w+= 1
# print(cat_dicts)
        

    