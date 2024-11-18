import os
import numpy as np
from random import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from  dataloader.Antidrone_dataloader import AntidronLoader
from nets.AFDTI import FUsion_net
from utils_pytorch.loss import _neg_loss

torch.manual_seed(42)
np.random.seed(42)

dataset = 1
dark_aug = 1
audio_seq = 1
image_seq = 0
workers = 4

batchsize = 32
dropout_rate=0.3
kerel_num=16
feature_dim=256

use_attention =0
Epoch = 200

checkpoint_path  = ''
print(use_attention,dark_aug,audio_seq,image_seq)

annotation_lines_train = '/home/kemove/yyz/Data/anotation_split_50/train.txt'
annotation_lines_val = '/home/kemove/yyz/Data/anotation_split_50/val.txt'
audio_path       = '/home/kemove/yyz/Data/np_data_align'
image_path       = '/home/kemove/yyz/Data/image'
detect_path      = '/home/kemove/yyz/Data/Detection_new'
gt_path          = '/home/kemove/yyz/Data/label'
num_class=6


train_name_all = annotation_lines_train.split("/")
train_name     = train_name_all[-2]
save_path        = 'output/'

save_name        = "{}_{}_{}_{}_{}_{}_heatmap".format(train_name,dataset,audio_seq,image_seq,use_attention,dark_aug)
save_path = os.path.join(save_path,save_name)
os.makedirs(save_path,exist_ok=True)


device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

with open(annotation_lines_train, "r") as f:
    train_lines = f.readlines()

with open(annotation_lines_val, "r") as f:
    val_lines = f.readlines()

shuffle(train_lines)
shuffle(val_lines)

num_train = len(train_lines)
num_val = len(val_lines)
epoch_step = num_train // batchsize
epoch_step_val = num_val // batchsize

train_data = AntidronLoader(train_lines ,audio_path,image_path,detect_path,gt_path,dark_aug=dark_aug,audio_seq=audio_seq,image_seq=image_seq,conv_2d=7)
val_data = AntidronLoader(val_lines ,audio_path,image_path,detect_path,gt_path,dark_aug=dark_aug,audio_seq=audio_seq,image_seq=image_seq,conv_2d=7)
train_dataloader = DataLoader(train_data,batchsize,shuffle=True,num_workers=workers,drop_last=True)
val_dataloader   = DataLoader(val_data,batchsize,shuffle=True,num_workers=workers,drop_last=True)

model = FUsion_net(use_attention=use_attention)
model = model.to(device)

if checkpoint_path!='':
    model.load_state_dict(torch.load(checkpoint_path))

optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
mse_loss  = torch.nn.MSELoss()
cross_entropy_loss = torch.nn.CrossEntropyLoss()


best_loss = 100
val_min = 100000
neg_loss = _neg_loss

for epoch in range(Epoch):
    model = model.train()
    with tqdm(total=epoch_step, unit='batch') as pbar:
        total_loss = 0
        for i, data in enumerate(train_dataloader, 0):
            spec,image,heatmap,diff,height,detect,cls,traj = data
            spec,image,heatmap,diff,height,detect,cls,traj = spec.to(device),image.to(device),heatmap.to(device),diff.to(device),height.to(device),detect.to(device),cls.to(device),traj.to(device)
            optimizer.zero_grad()
            mask_heat = (torch.max(heatmap.view(heatmap.size(0), -1), dim=1).values != 0).float().view(-1, 1, 1)
            mask_traj = (traj != 0).float()
            mask_z    = (height!=0).float()
            mask_off  = (diff!=0).float()
            h,z,o,d,t,c = model(spec,image)
            loss_cls   = cross_entropy_loss(c,cls)
            loss_detect      = cross_entropy_loss(d,detect)
            loss_traj        = mse_loss(t*mask_traj,traj*mask_traj)
            loss_z           = mse_loss(z*mask_z,height*mask_z)
            loss_heatmap     = neg_loss(h*mask_heat,heatmap*mask_heat)
            loss_off           = mse_loss(o*mask_off,diff*mask_off)
            loss =  loss_heatmap+loss_z+loss_off*0.5+loss_detect*0.3+loss_traj*0.3+loss_cls*0.3
            total_loss = total_loss+loss.item()
            loss.backward()
            optimizer.step()
            pbar.update(1)
        print(total_loss/len(train_dataloader))

        if epoch % 1 == 0:
            model = model.eval()
            total_loss = 0
            with torch.no_grad():
                for i, data in enumerate(val_dataloader, 0):
                    spec,image,heatmap,diff,z,detect,cls,traj = data
                    spec,image,heatmap,diff,height,detect,cls,traj = spec.to(device),image.to(device),heatmap.to(device),diff.to(device),z.to(device),detect.to(device),cls.to(device),traj.to(device)

                    mask_traj = (traj != 0).float()
                    mask_heat = (torch.max(heatmap.view(heatmap.size(0), -1), dim=1).values != 0).float().view(-1, 1, 1)
                    mask_z    = (height!=0).float()
                    mask_off  = (diff!=0).float()
                    h,z,o,d,t,c = model(spec,image)
                    loss_cls   = cross_entropy_loss(c,cls)
                    loss_detect      = cross_entropy_loss(d,detect)
                    loss_traj        = mse_loss(t*mask_traj,traj*mask_traj)
                    loss_z           = mse_loss(z*mask_z,height*mask_z)
                    loss_heatmap     = neg_loss(h*mask_heat,heatmap*mask_heat)
                    loss_off           = mse_loss(o*mask_off,diff*mask_off)
                    loss =  loss_heatmap+loss_z+loss_off*0.5+loss_detect*0.3+loss_traj*0.3+loss_cls*0.3

                    total_loss = total_loss+loss.item()

            total_loss = total_loss/len(val_dataloader)

        print(total_loss)

        if total_loss<=val_min:
            val_min = total_loss
            torch.save(model.state_dict(), '%s/%s' % (save_path,"best_epoch"))
        if epoch % 1 == 0:
            torch.save(model.state_dict(), '%s/%s' % (save_path,"last_epoch"))
