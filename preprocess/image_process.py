import os.path
import random
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import cv2


def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return rst


def image_darkaug(img,img_label,dark_aug,brightness=1,conv2d=0):
    if dark_aug==1:
        creterion = random.random()
        if creterion>0.5:
            brightness = 0
            img = imgBrightness(img,brightness,3)
            img_label = np.array([0])

    elif dark_aug==2:
            img = imgBrightness(img,brightness,3)
            img_label = np.array([0])

    elif dark_aug>2:
        brightness = 0.04/dark_aug
        img = imgBrightness(img,brightness,3)
        img_label = np.array([0])

    img = preprocess_input(img)
    return img,img_label,brightness

def preprocess_input(image):
    image = image / 127.5-1
    return image


def make_class(name):
    parts = name.split("/")
    drone_name = parts[0]
    if drone_name=='m3e' or drone_name=='b1' :
        cls = np.array([0])
    elif drone_name=='m30s1' or drone_name=='m30s2' or drone_name=='b2':
        cls = np.array([1])
    elif drone_name=='m300s1' or drone_name=='m300s2' or drone_name=='b3':
        cls = np.array([2])
    elif drone_name=='p4s1' or drone_name=='p4s2' or drone_name=='b4':
        cls = np.array([3])
    elif drone_name=='b5':
        cls = np.array([4])
    else:
        cls =np.array([5])
    return cls


def make_traj(gt_path,name):
    parts = name.split("/")
    index = parts[-1][:-4]
    future_traj = np.load(os.path.join(gt_path,name)).reshape(1,3)
    for f in range(1,10):
        if len(parts)==2:
            file_name   =  f"{parts[0]}/{int(index)+f}.npy"
            current_pos = np.load(os.path.join(gt_path,file_name)).reshape(1,3)
            future_traj = np.concatenate((future_traj,current_pos),0)
        else:
            file_name   =  f"{parts[0]}/{parts[1]}/{int(index)+f}.npy"
            current_pos = np.load(os.path.join(gt_path,file_name)).reshape(1,3)
            future_traj = np.concatenate((future_traj,current_pos),0)
    return future_traj

def make_img_seq(image_path,name,image,brightness):
    parts = name.split("/")
    index = parts[-1][:-4]
    past_image = image[np.newaxis,...]
    for f in range(1,4):
        if len(parts)==2:
            file_name   =  f"{parts[0]}/{int(index)-2*f}.png"
            current_image_name = os.path.join(image_path,file_name)
            current_image  = cv2.imread(current_image_name,cv2.IMREAD_COLOR)[:,:1280,:]
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            current_image,_,_ = image_darkaug(current_image,np.array([1]),2,brightness)
            current_image = current_image[np.newaxis,...]
            past_image = np.concatenate([past_image,current_image],axis=0)
        if len(parts)==3:
            file_name   =  f"{parts[0]}/{parts[1]}/{int(index)-2*f}.png"
            current_image_name = os.path.join(image_path,file_name)
            current_image  = cv2.imread(current_image_name,cv2.IMREAD_COLOR)[:,:1280,:]
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            current_image,_,_ = image_darkaug(current_image,np.array([1]),2,brightness)
            current_image = current_image[np.newaxis,...]
            past_image = np.concatenate([past_image,current_image],axis=0)
    return past_image
