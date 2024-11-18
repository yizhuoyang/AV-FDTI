import numpy as np
import os


output_dir = '/home/kemove/yyz/Data/class'
ref_dir    = '/home/kemove/yyz/Data/label'

bag_list  = os.listdir(ref_dir)




for bag_name in bag_list:

    if bag_name=='b1':
        cls = np.array([0])
    elif bag_name=='b2':
        cls = np.array([1])
    elif bag_name=='b3':
        cls = np.array([2])
    elif bag_name=='b4':
        cls = np.array([3])
    elif bag_name=='b5':
        cls = np.array([4])
    else:
        cls = np.array([5])

    output_bag_dir = os.path.join(output_dir,bag_name)
    os.makedirs(output_bag_dir,exist_ok=True)
    ref_bag_dir    = os.path.join(ref_dir,bag_name)
    label_list     = os.listdir(ref_bag_dir)

    for i in range(len(label_list)):
        np.save(output_bag_dir+'/'+str(i)+'.npy',cls)

