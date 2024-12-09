import os
from preprocess.audio_process import *
from preprocess.image_process import *
import cv2
import numpy as np
import torch
from preprocess.audio_process import make_seq_audio, Audio2Spectrogram
from preprocess.image_process import image_darkaug, make_traj, make_class
from utils.metrics import calculate_result
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import transforms as trans
def load_test(data,audio_seq,image_path,gt_path,audio_path,detect_path,dark_aug):
    data = data[:-4]+'npy'
    audio_name  = os.path.join(audio_path,data[:-4]+'.npy')
    image_name = os.path.join(image_path,data[:-4]+'.png')
    gt_name     = os.path.join(gt_path,data[:-4]+'.npy')

    if audio_seq:
        audio   = make_seq_audio(audio_path,data)
    else:
        audio   = np.load(audio_name)

    audio   = np.transpose(audio,[1,0])
    spec       = Audio2Spectrogram(audio,sr=48000,min_frequency=1000,max_frequency=8000)
    detect_name =  os.path.join(detect_path,data[:-4]+'.npy')
    detect      = np.load(detect_name)

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

    traj    = make_traj(gt_path,data[:-4]+'.npy')
    traj    = torch.from_numpy(traj).float()

    return spec,image,gt,detect,cls,traj


def load_annotations(annotation_file):
    with open(annotation_file, "r") as f:
        lines = f.readlines()
    return lines


def evaluate_model(model, val_lines, config):
    """
    Evaluate the model on validation data with tqdm progress bar.

    Args:
        model: The PyTorch model to evaluate.
        val_lines: List of validation data samples.
        config: Dictionary containing configuration parameters.

    Returns:
        gt_class: List of ground truth classes.
        predict_class: List of predicted classes.
        real_positions: List of ground truth positions.
        predicted_positions: List of predicted positions.
        class_right: Number of correctly classified samples.
    """
    device = config["device"]
    audio_seq = config["audio_seq"]
    dark_aug = config["dark_aug"]
    audio_path, image_path = config["audio_path"], config["image_path"]
    detect_path, gt_path = config["detect_path"], config["gt_path"]

    class_right = 0
    gt_class, predict_class = [], []
    real_positions, predicted_positions = [], []

    # Use tqdm to display progress
    with tqdm(total=len(val_lines), desc="Evaluating", unit="sample") as pbar:
        for data in val_lines:
            spec, image, gt, detect, cls, traj = load_test(data, audio_seq, image_path, gt_path, audio_path, detect_path, dark_aug)
            real_class = np.array(cls)
            real_position = np.array(gt)
            gt_class.append(real_class)

            # Image resizing
            resize_transform = trans.Resize((256, 256), antialias=True)
            image = resize_transform(image)

            spec, image, gt, detect, cls, traj = spec.to(device), image.to(device), gt.to(device), detect.to(device), cls.to(device), traj.to(device)
            h, z, o, d, t, c = model(spec.unsqueeze(0), image.unsqueeze(0))

            h = h.cpu().detach().numpy()[0]
            class_detection = c.cpu().detach().numpy()[0]
            o = o.cpu().detach().numpy()[0]
            z = z.cpu().detach().numpy()[0]

            predicted_position = calculate_result(h, o, z)
            real_positions.append(real_position)
            predicted_positions.append(predicted_position)

            predicted_cls = np.argmax(class_detection)
            predict_class.append(predicted_cls)

            if predicted_cls == real_class:
                class_right += 1

            # Update progress bar
            pbar.update(1)

    return gt_class, predict_class, real_positions, predicted_positions, class_right

def calculate_metrics(gt_class, predict_class, real_positions, predicted_positions, config):
    # Classification accuracy
    accuracy = sum(np.array(gt_class) == np.array(predict_class)) / len(gt_class)
    print(f"The classification accuracy is: {accuracy}")

    # Position error
    real_positions = np.array(real_positions)
    predicted_positions = np.array(predicted_positions)
    position_error = np.linalg.norm(real_positions - predicted_positions, axis=1)
    mean_position_error = np.mean(position_error)
    print(f"Mean position error: {mean_position_error}")

    # Distance in each dimension
    abs_distances = np.abs(real_positions - predicted_positions)
    x_distances, y_distances, z_distances = abs_distances[:, 0], abs_distances[:, 1], abs_distances[:, 2]
    print(f"Mean distances (x, y, z): {x_distances.mean()}, {y_distances.mean()}, {z_distances.mean()}")

    return accuracy, mean_position_error


def plot_confusion_matrix(gt_class, predict_class, config):
    class_labels = ['Drone1', 'Drone2', 'Drone3', 'Drone4', 'Drone5', 'None']
    cm = confusion_matrix(gt_class, predict_class)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, cbar=False)
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('Ground Truth', fontsize=20)
    plt.savefig(config["confusion_matrix_path"], dpi=500, transparent=True, bbox_inches='tight', pad_inches=0.1, format='png')
    plt.show()
