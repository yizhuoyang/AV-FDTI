from torch.utils.data.dataset import Dataset
from preprocess.audio_process import *
from preprocess.image_process import *
import torchvision.transforms as trans

np.random.seed(42)

class AntidroneLoader(Dataset):
    def __init__(self, annotation_lines,audio_path,image_path,detect_path,gt_path,frequency=[1000,8000],dark_aug=0,audio_seq=0,network_type=0):
        super(AntidroneLoader, self).__init__()
        self.annotation_lines   = annotation_lines
        self.audio_path         = audio_path
        self.image_path         = image_path
        self.gt_path            = gt_path
        self.frequency          = frequency
        self.dark_aug           = dark_aug
        self.detect_path        = detect_path
        self.audio_seq          = audio_seq
        self.network_type       = network_type

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
        spec       = Audio2Spectrogram(audio,sr=48000,min_frequency=self.frequency[0],max_frequency=self.frequency[1])
        spec       = spec.float()
        detect      = np.load(detect_name)

        image  = cv2.imread(image_name,cv2.IMREAD_COLOR)[:,:1280,:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image,detect,brightness  = image_darkaug(image,detect,self.dark_aug)
        image         = np.transpose(image,[2,0,1])
        image         = torch.from_numpy(image).float()

        if self.network_type==1:            # This is the input for vora's network which required audio spectrogram in shape (224,224) (w*h)
            resize_transform = trans.Resize((256,256),antialias=True)
            image  = resize_transform(image)
            resize_transform = trans.Resize((224,224),antialias=True)
            spec = resize_transform(spec)
        elif self.network_type==2:          # This is the input for talknet
            resize_transform = trans.Resize((112,112),antialias=True)
            image  = resize_transform(image)
            resize_transform = trans.Resize((13,13),antialias=True)
            spec = resize_transform(spec)
        else:                               # This is the input for darknet, visualnet, audionet, avped
            resize_transform = trans.Resize((256,256),antialias=True)
            image  = resize_transform(image)

        detect = detect[0]
        detect = torch.tensor(detect)

        gt      = np.load(gt_name)
        gt     = torch.from_numpy(gt).float()

        cls      = make_class(name)
        cls      = cls[0]
        cls = torch.tensor(cls)

        traj    = make_traj(self.gt_path,name[:-4]+'npy')
        traj    = torch.from_numpy(traj).float()

        return spec,image,gt,detect,cls,traj
