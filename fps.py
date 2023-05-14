import cv2

cam = cv2.VideoCapture("/Users/kalebnew/Desktop/obj_states_scratch/uncut_vids/tie_shoe.mp4")

fps = cam.get(cv2.CAP_PROP_FPS)
print(fps)

n= 0
i=0
while True:
    ret, frame = cam.read()
    if n%fps ==0:
        cv2.imwrite("{}.jpg".format(i), frame)
        i += 1
    
    n+=1
    if ret == False:
        break;
cam.release()
cv2.destroyAllWindows()