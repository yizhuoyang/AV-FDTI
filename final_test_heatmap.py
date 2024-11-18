import numpy as np
from random import *
import torch
from matplotlib import pyplot as plt
from preprocess.audio_process import *
from preprocess.image_process import *
from preprocess import *
from utils_pytorch.metrics import calculate_result
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils_pytorch.test_loader import load_test
from nets.AFDTI import FUsion_net
torch.manual_seed(42)
np.random.seed(42)



dataset = 1
dark_aug = 0
audio_seq = 1
image_seq = 0
workers = 8

batchsize = 32
dropout_rate=0.3
kerel_num=32
feature_dim=256
use_attention =4

use_tracking = 0

checkpoint_path  = '/home/kemove/yyz/AV-FDTI/output/anotation_split_50_1_1_0_4_1_heatmap/last_epoch'

if dataset==0:
    annotation_lines_train = '/home/kemove/yyz/MMAUD/challenge_data2/data_split/train.txt'
    annotation_lines_val = '/home/kemove/yyz/MMAUD/challenge_data2/data_split/val.txt'
    audio_path       = '/home/kemove/yyz/MMAUD/challenge_data2/audio_numpy'
    image_path       = '/home/kemove/yyz/MMAUD/challenge_data2/Image'
    detect_path      = '/home/kemove/yyz/MMAUD/challenge_data2/detect_new'
    gt_path          = '/home/kemove/yyz/MMAUD/challenge_data2/ground_truth'
    num_class=4


else:
    annotation_lines_train = '/home/kemove/yyz/Data/anotation_split_50/train.txt'
    annotation_lines_val = '/home/kemove/yyz/Data/anotation_split_50/val.txt'
    audio_path       = '/home/kemove/yyz/Data/np_data_align'
    image_path       = '/home/kemove/yyz/Data/image'
    detect_path      = '/home/kemove/yyz/Data/Detection_new'
    gt_path          = '/home/kemove/yyz/Data/label'
    num_class=6

device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(annotation_lines_train, "r") as f:
    train_lines = f.readlines()

with open(annotation_lines_val, "r") as f:
    val_lines = f.readlines()

model = FUsion_net(use_attention=use_attention)
model = model.to(device)
model = model.eval()
if checkpoint_path!='':
    model.load_state_dict(torch.load(checkpoint_path))

class_right = 0
r = []
l = []
bad_data = []

gt_class = []
predict_class = []


for data in val_lines:

    spec,image,gt,detect,cls,traj = load_test(data,audio_seq,image_seq,image_path,gt_path,audio_path,detect_path,dark_aug)
    real_class = np.array(cls)
    real_position = np.array(gt)
    gt_class.append(real_class)
    resize_transform = trans.Resize((256,256),antialias=True)
    image  = resize_transform(image)

    spec,image,gt,detect,cls,traj = spec.to(device),image.to(device),gt.to(device),detect.to(device),cls.to(device),traj.to(device)

    h,z,o,d,t,c  = model(spec.unsqueeze(0),image.unsqueeze(0))

    h = h.cpu().detach().numpy()[0]
    class_detection = c.cpu().detach().numpy()[0]
    trajectory      = t.cpu().detach().numpy()[0]
    o = o.cpu().detach().numpy()[0]
    z = z.cpu().detach().numpy()[0]

    position = calculate_result(h,o,z)
    print(position,real_position)


    if real_class!=np.array([5]):
        l.append(position)
        r.append(real_position)

    print(np.mean(np.linalg.norm(np.array(r)-np.array(l),axis=1)))

    predicted_cls = np.argmax(class_detection)
    predict_class.append(predicted_cls)

    if predicted_cls==real_class:
        class_right+=1



print("The classification acc is: {}".format(class_right/len(val_lines)))

r,l = np.array(r),np.array(l)
error = np.linalg.norm(r - l,axis=1)
error = np.mean(error)
print(error)
abs_distance = np.abs(r-l)
x_distances = abs_distance[:, 0]  # Distances in the x dimension
y_distances = abs_distance[:, 1]  # Distances in the y dimension
z_distances = abs_distance[:, 2]  # Distances in the z dimension
print(x_distances.mean(),y_distances.mean(),z_distances.mean())
predict_class = np.array(predict_class)
gt_class = np.array(gt_class)


fs = 25
cm = confusion_matrix(gt_class, predict_class)

# Define class labels (replace with your actual class labels)
class_labels = ['Drone1','Drone2','Drone3','Drone4', 'Drone5', 'None']
plt.figure(figsize=(8, 8))
ax = plt.gca()

# Increase the font size for the 'Class 0' label
for i in range(6):
    ax.get_xticklabels()[i].set_fontsize(14)
    ax.get_yticklabels()[i].set_fontsize(14)
# Create a confusion matrix heatmap using seaborn
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": fs },cbar=False)
plt.xlabel('Predicted', fontsize=20 )
plt.ylabel('Ground Truth', fontsize=20 )

plt.savefig('confusion_matrix_d.png' ,dpi=500, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')

plt.show()
