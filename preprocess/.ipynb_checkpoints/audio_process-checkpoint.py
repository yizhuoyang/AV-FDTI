import os
from random import random
import librosa
import matplotlib.pyplot as plt
import matplotlib
import torch


matplotlib.use('TKAgg')
from scipy import signal
import numpy as np
from scipy.signal import butter, lfilter
import torchaudio.transforms as T
import torchvision.transforms as trans

def normalization_processing(data):

    data_min = data.min()
    data_max = data.max()

    data = data - data_min
    data = data / (data_max-data_min)

    return data

def normalization_processing_torch(data):
    # Assuming data is a PyTorch tensor
    data_min = torch.min(data)
    data_max = torch.max(data)

    # Normalizing the data
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data

def normalization_processing_torch_all(data):
    for i in range(data.shape[0]):
        data[i,:] = normalization_processing_torch(data[i,:])
    return data

def save_bandpass_audio(input_path, lowcut, highcut, sr=None):
    # Load the audio file
    y, sr = librosa.load(input_path, sr=sr)

    # Define the band-pass filter
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(N=4, Wn=[low, high], btype='band', output='sos')

    # Apply the band-pass filter to the audio signal
    filtered_audio = signal.sosfilt(sos, y)

    return filtered_audio
    # Save the filtered audio to a new file using soundfile
    #sf.write(output_path, filtered_audio, sr)
    #save_bandpass_audio(input_file, output_file, lowcut, highcut)


def Audio2Spectrogram(np_data,sr,num_audio=6,normarlization=1,min_frequency=8000,max_frequency=10000,eliminate=0,conv_2d=0):

    np_data   = torch.tensor(np_data,dtype=torch.float32)

    melspectrogram = T.MelSpectrogram(
        sample_rate = sr,
        n_fft = 2048,
        hop_length=512,
        n_mels=20,
        f_min=min_frequency,
        f_max=max_frequency,
        pad_mode='constant',
        norm='slaney',
        mel_scale='slaney',
        power=2,

    )
    spectrogram = melspectrogram(np_data)
    # print(spectrogram.shape)
    if normarlization!=0:
        # spectrogram = spectrogram/np.linalg.norm(spectrogram,axis=0,keepdims=True)
        spectrogram = normalization_processing_torch_all(spectrogram)

    if conv_2d==0 or conv_2d==7:
        resize_transform = trans.Resize((64,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
    elif conv_2d==1:
        resize_transform = trans.Resize((256,256),antialias=True)
        spectrogram = resize_transform(spectrogram)
    elif conv_2d ==100:
        resize_transform = trans.Resize((13,13),antialias=True)
        spectrogram = resize_transform(spectrogram)
    else:
        resize_transform = trans.Resize((224,224),antialias=True)
        spectrogram = resize_transform(spectrogram)

    return spectrogram


def process_audio(original_audio):

    mean_value = np.mean(np.abs(original_audio[0]))
    high_value_indices = np.where(np.abs(original_audio[0]) > 2*mean_value)[0]
    if len(high_value_indices) > 0:
        first_high_index = high_value_indices[0]

        if first_high_index + 2205 < len(original_audio[0]):
            for i in range(8):
                original_audio[i][first_high_index: first_high_index +2205] = 0
        else:
            for i in range(8):
                original_audio[i][first_high_index:] = 0
            print('ok')

    return original_audio

def make_seq_audio(audio_path,name):
    parts = name.split("/")
    index = parts[-1][:-4]
    past_audio = np.load(os.path.join(audio_path,name))
    for f in range(1,10):
        if len(parts)==2:
            file_name   =  f"{parts[0]}/{int(index)-f}.npy"
            current_pos = np.load(os.path.join(audio_path,file_name))
            past_audio = np.concatenate((past_audio,current_pos),0)
        if len(parts)==3:
            file_name   =  f"{parts[0]}/{parts[1]}/{int(index)-f}.npy"
            current_pos = np.load(os.path.join(audio_path,file_name))
            past_audio = np.concatenate((past_audio,current_pos),0)
    return past_audio


def obtain_gaussian2(size,current_position,radius=40, k=1):
    x_size, y_size =size[0],size[1]
    center   = current_position[:2]
    mean_x,mean_y      = center[0],center[1]

    x_len = 5.861056694237837+1.827702940235904
    y_len = 18.766594286932648+4.720740577048357
    ratiox = x_size/x_len
    ratioy = y_size/y_len

    Tmean_x = np.round((mean_x+1.827702940235904)*x_size/x_len,3)
    Tmean_y = np.round((mean_y+4.720740577048357)*y_size/y_len,3)

    heatmap  = np.zeros((x_size,y_size), dtype=np.float32)
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(Tmean_x), int(Tmean_y)

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    diff = np.array([(Tmean_x-x)*100/ratiox,(Tmean_y-y)*100/ratioy])

    return heatmap,diff

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
