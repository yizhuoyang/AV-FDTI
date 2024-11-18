import os
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
from preprocess.audio_process import *
from preprocess.image_process import *

np.random.seed(42)

class AntidronLoader(Dataset):
    def __init__(self, annotation_lines,audio_path,image_path,detect_path,gt_path,frequency=[1000,8000],conv_2d=0,dark_aug=0,audio_seq=0,image_seq=0):
        super(AntidronLoader, self).__init__()
        self.annotation_lines   = annotation_lines
        self.audio_path         = audio_path
        self.image_path         = image_path
        self.gt_path            = gt_path
        self.frequency          = frequency
        self.conv_2d            = conv_2d
        self.dark_aug           = dark_aug
        self.detect_path        = detect_path
        self.audio_seq          = audio_seq
        self.image_seq          = image_seq


    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):

        name       = self.annotation_lines[index]
        audio_name  = os.path.join(self.audio_path,name[:-4]+'npy')
        image_name  = os.path.join(self.image_path,name[:-4]+'png')
        gt_name     = os.path.join(self.gt_path,name[:-4]+'npy')
        detect_name =  os.path.join(self.detect_path,name[:-4]+'npy')


        if self.audio_seq:
            audio   = make_seq_audio(self.audio_path,name[:-4]+'npy')
        else:
            audio   = np.load(audio_name[:])
        audio   = np.transpose(audio,[1,0])
        spec       = Audio2Spectrogram(audio,sr=48000,min_frequency=self.frequency[0],max_frequency=self.frequency[1],conv_2d=self.conv_2d)
        spec       = spec.float()
        detect      = np.load(detect_name)

        if self.image_seq:
            image  = cv2.imread(image_name,cv2.IMREAD_COLOR)[:,:1280,:]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image,detect,brightness  = image_darkaug(image,detect,self.dark_aug)
            image = make_img_seq(self.image_path,name[:-1],image,brightness)
            image         = np.transpose(image,[0,3,1,2])

        else:
            image  = cv2.imread(image_name,cv2.IMREAD_COLOR)[:,:1280,:]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image,detect,brightness  = image_darkaug(image,detect,self.dark_aug,self.conv_2d)
            image         = np.transpose(image,[2,0,1])

        image         = torch.from_numpy(image).float()


        if self.conv_2d==100:
            resize_transform = trans.Resize((112,112),antialias=True)
            image  = resize_transform(image)
        else:
            resize_transform = trans.Resize((256,256),antialias=True)
            image  = resize_transform(image)

        detect = detect[0]
        detect = torch.tensor(detect)


        if self.conv_2d!=7:
            gt      = np.load(gt_name)
            gt     = torch.from_numpy(gt).float()
        else:
            gt      = np.load(gt_name)
            heatmap,diff     = obtain_gaussian2([512,512],gt.astype(np.float32))
            z = gt[-1:]
            heatmap = torch.from_numpy(heatmap).float()
            z       = torch.from_numpy(z).float()
            diff    = torch.from_numpy(diff).float()

        cls      = make_class(name)
        cls      = cls[0]
        cls = torch.tensor(cls)

        traj    = make_traj(self.gt_path,name[:-4]+'npy')
        traj    = torch.from_numpy(traj).float()

        if self.conv_2d!=7:
            return spec,image,gt,detect,cls,traj

        else:
            return spec,image,heatmap,diff,z,detect,cls,traj
