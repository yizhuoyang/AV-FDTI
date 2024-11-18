import os
from preprocess.audio_process import *
from preprocess.image_process import *
import cv2
import numpy as np
import torch
from preprocess.audio_process import make_seq_audio, Audio2Spectrogram
from preprocess.image_process import image_darkaug, make_img_seq, make_traj, make_class


def load_test(data,audio_seq,image_seq,image_path,gt_path,audio_path,detect_path,dark_aug):
    data = data[:-4]+'npy'
    audio_name  = os.path.join(audio_path,data[:-4]+'.npy')
    image_name = os.path.join(image_path,data[:-4]+'.png')
    gt_name     = os.path.join(gt_path,data[:-4]+'.npy')

    if audio_seq:
        audio   = make_seq_audio(audio_path,data)
    else:
        audio   = np.load(audio_name)

    audio   = np.transpose(audio,[1,0])
    spec       = Audio2Spectrogram(audio,sr=48000,min_frequency=1000,max_frequency=8000,conv_2d=0)
    detect_name =  os.path.join(detect_path,data[:-4]+'.npy')
    detect      = np.load(detect_name)

    if image_seq:
        image  = cv2.imread(image_name,cv2.IMREAD_COLOR)[:,:1280,:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image,detect,brightness  = image_darkaug(image,detect,dark_aug)
        image = make_img_seq(image_path,data[:-1],image,brightness)
        image         = np.transpose(image,[0,3,1,2])

    else:
        image  = cv2.imread(image_name,cv2.IMREAD_COLOR)[:,:1280,:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image,detect,brightness  = image_darkaug(image,detect,dark_aug)
        image         = np.transpose(image,[2,0,1])



    image         = torch.from_numpy(image).float()
    #
    resize_transform = trans.Resize((256,256),antialias=True)
    image  = resize_transform(image)

    detect = detect[0]
    detect = torch.tensor(detect)

    gt      = np.load(gt_name)
    gt     = torch.from_numpy(gt).float()

    cls      = make_class(data)
    cls      = cls[0]
    cls = torch.tensor(cls)
    # cls     = torch.from_numpy(cls).float()

    traj    = make_traj(gt_path,data[:-4]+'.npy')
    traj    = torch.from_numpy(traj).float()

    return spec,image,gt,detect,cls,traj
