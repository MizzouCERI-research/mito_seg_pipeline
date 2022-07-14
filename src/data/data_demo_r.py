
# %%
# region IMPORT ###########################################
###########################################################

from IPython import get_ipython
from torch import int64
from torch._C import dtype

# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# python.dataScience.interactiveWindowMode


seed = 1

import os
import sys
local_path = '../../' # Jupyter
# local_path = './' # Terminal
sys.path.append(local_path)
# sys.path.append(os.getcwd())
import glob
import random
random.seed(seed)
# from os import path, listdir

import numpy as np
# np.random.seed(seed)
# numpy.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=np.inf)
# pd.random_state = seed
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', 0)
# pd.set_option('display.max_columns', 0)

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.ioff()
# get_ipython().run_line_magic('matplotlib', 'agg')
# get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn-dark')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")



import scipy.io as sio
# import mat73
import hdf5storage

from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score, precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight



import torch
# from torch.utils import data as D
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import torchvision
# import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torchvision.transforms.functional as Ft
# import torch.nn.functional as Fn


# from torch.optim import lr_scheduler
# from torchsummary import summary


# from monai.data import PatchDataset, DataLoader
from monai.transforms import RandSpatialCropSamplesd


# import sys 
# sys.path.append('..') # Jupyter
# from visualization.viz_torchio import plot_volume_interactive # Jupyter
# from ..visualization.viz_torchio import plot_volume_interactive # not work
# from ...src.visualization.viz_torchio import plot_volume_interactive # not work
# from src.visualization.viz_torchio import plot_volume_interactive # Terminal

from src.visualization.viz_torchio import plot_volume_interactive # Terminal


import monai
from monai.data import Dataset
# from monai.utils import get_seed, progress_bar
# from monai.networks.layers import Norm
from monai.transforms import \
    apply_transform, Compose, Randomizable, Transform, \
    ToTensord, AddChanneld,  \
    Resized, Resize, CropForegroundd, RandCropByPosNegLabeld, RandSpatialCropd, \
    RandRotate90d, Orientationd, RandAffined, Spacingd, RandRotated, RandZoomd, Rand3DElasticd, RandAxisFlipd, \
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd, ScaleIntensityd, ScaleIntensityRanged, RandShiftIntensityd


from src2.connectomics.data.utils.data_segmentation import seg_widen_border, seg_to_instance_bd,  seg_to_weights
from src2.connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from src2.connectomics.config import get_cfg_defaults, save_all_cfg, update_inference_cfg


# endregion IMPORT






# %%
# region DATASET CLASS ####################################
###########################################################
class PatchDataset(Dataset):
    """
    This custom dataset class take root directory and train flag, 
    and return dataset training dataset id train flag is true 
    else is return validation dataset.
    """
    
    def __init__(self, data_root, data_list, set_type=0, n_patches=5, patch_size=(256,256), image_shape=None, transform=None, transform_patch=None):
        
        """
        init method of the class.
        
         Parameters:
         
         data_root (string): path of root directory.
         
         train (boolean): True for training dataset and False for test dataset.
         
         image_shape (int or tuple or list): [optional] int or tuple or list. Defaut is None. 
                                             It is not None image will resize to the given shape.
                                 
         transform (method): method that will take PIL image and transforms it.
         
        """
        
        # basic data config
        self.data_root = data_root
        self.data_list = data_list
        self.set_type = set_type
        self.num_classes = 1
        

        # image patch sampler
        self.num_per_image = n_patches
        self.patch_size = np.array(patch_size)
        if self.num_per_image > 0:
            self.sampler = RandSpatialCropSamplesd(keys=['image', 'label'], roi_size=self.patch_size, num_samples=self.num_per_image,
                                        random_center=True, random_size=False)
        else:
            self.sampler = None

        # set image_resize attribute
        if image_shape is not None:
            if isinstance(image_shape, int):
                self.image_shape = (image_shape, image_shape)
            
            elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
                assert len(image_shape) == 1 or len(image_shape) == 2, 'Invalid image_shape tuple size'
                if len(image_shape) == 1:
                    self.image_shape = (image_shape[0], image_shape[0])
                else:
                    self.image_shape = image_shape
            else:
                raise NotImplementedError 
                
        else:
            self.image_shape = image_shape
            
        # set transform attribute
        self.transform = transform
        self.transform_patch = transform_patch

        # 
        # topt = '4-2-1'
        # topt[0] == '4': # instance boundary mask
        #  _, bd_sz,do_bg = [int(x) for x in topt.split('-')]
        self.LABEL_EROSION = 1
        self.boundary_size = 2 # bd_sz
        self.do_background = 1 # do_bg

        # store im info
        self.pre_im_id = None
        self.pre_im_file = None
        # self.data = [None for i in range(len(self.data_list))]
        self.data = []



    
    def read_full_image(self, image_id):
        # Read full image data and store into the self.data dictionary
        image_file = os.path.normpath(self.data_list[image_id])
        # file_parts = image_file.split('\\')
        file_parts = os.path.split(image_file)

        # image = sio.loadmat(image_file)
        # image = image['outImg']
        
        if self.set_type != 2: # load label if having annotation (not test data)
            # image = sio.loadmat(image_file)
            image = hdf5storage.loadmat(image_file)
            image = image['image']
            
            
            label_file = image_file.replace('images', 'labels').replace('image_', 'seg_')
            # label = sio.loadmat(label_file)
            label = hdf5storage.loadmat(label_file)
            label = label['label']

            # shell_file = label_file[:-4] + '_boundary.mat'
            shell_file = label_file
            shell = hdf5storage.loadmat(shell_file)
            # shell = shell['out_shell']
            shell = shell['label']


            location_file = label_file.replace('.mat', '.txt')
            locations = []
            # # class_names = ['4V94',	'4CR2', '1QVR', '1BXN', '3CF3', '1U6G', '3D2F', '2CG9', '3H84', '3GL1', '3QM1', '1S3X', '5MRC', 'vesicle', 'fiducial']
            # with open(location_file, 'r') as f:
            #     for line in f:
            #         # pdb_id, X, Y, Z, *_ = line.rstrip('\n').split()
            #         X, Y, Z, = line.rstrip('\n').split()

            #         # idx = class_names.index(pdb_id)
            #         # if idx == 11:
            #         #     prob = 1
            #         # elif idx in [5, 6, 7, 8, 9, 10]:
            #         #     prob = 0.8
            #         # elif idx == 13:
            #         #     prob = 0.3
            #         # else:
            #         #     # prob = 0.35
            #         #     prob = 0.6
            #         # locations.append((int(X), int(Y), int(Z), prob))
            #         locations.append((int(X), int(Y), int(Z)))
            # locations = np.array(locations)
            # # locations[:, 3] /= np.sum(locations[:, 3])

            locations = np.zeros((1, 3))
            
        else:
            # image = sio.loadmat(image_file)
            # image = mat73.loadmat(image_file)
            image = hdf5storage.loadmat(image_file)
            image = image['image']

            label = np.zeros_like(image)
            locations = np.zeros((1, 3))


        # # image = 1 - (image - np.min(image)) / (np.max(image) - np.min(image))
        # # image = 255 - image
        # # label = 1 - label
        # label[label>0] = 255
        # # label = np.array(label).astype(np.float32)


        im_min = np.min(image)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        image = image.astype(np.float32)
        # label_instances = label.copy()
        # label[label>0] = 1
        label = np.array(label).astype(np.float32)
        shell = np.array(shell).astype(np.float32)

        # print(image.shape)


        # image = Image.fromarray(image)
        # label = Image.fromarray(label)

        if self.image_shape is not None:    
            image = Ft.resize(image, self.image_shape)
            label = Ft.resize(label, self.image_shape)
            shell = Ft.resize(shell, self.image_shape)

        if self.transform is not None:
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            shell = np.expand_dims(shell, axis=0)
            image = self.transform(image).squeeze()
            label = self.transform(label).squeeze()
            shell = self.transform(shell).squeeze()
            # label = torch.tensor(np.array(label), dtype=torch.float32)
            # label = Ft.to_tensor(label)
            # shell = Ft.to_tensor(shell)
            

        if self.num_per_image > 0 or self.transform is None:
            image = np.array(image)
            label = np.array(label).astype(np.float32)
            shell = np.array(shell).astype(np.float32)

        # boundary
        # label_2 = seg_widen_border(label, self.LABEL_EROSION)
        # label_3 = seg_to_instance_bd(label, self.boundary_size, self.do_background).astype(np.float32)
        # labels = np.stack([label_2, label_3])

        # add channel dim
        if self.num_per_image > 0:
            while(len(image.shape) - len(self.patch_size) < 1): 
                image = np.expand_dims(image, axis=0)
                label = np.expand_dims(label, axis=0)
                shell = np.expand_dims(shell, axis=0)

            # boundary
            # while(len(labels.shape) - len(self.patch_size) < 1): 
            #     labels = np.expand_dims(labels, axis=0)
        else:
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            shell = np.expand_dims(shell, axis=0)

        
        instances = label.copy()
        label[label>0] = 1
        # label = np.array(label).astype(np.float32)


        self.pre_im_id = image_id
        self.pre_im_file = file_parts[-1]

        # self.data = {}
        # self.data['image'] = image
        # self.data['label'] = label
        # self.data['file_name'] = file_parts[-1]
        # self.data['id'] = image_id

        self.data.append({})
        # self.data[image_id] = {}
        self.data[image_id]['image'] = image
        self.data[image_id]['label'] = label
        # self.data[image_id]['labels'] = labels
        self.data[image_id]['shell'] = shell

        self.data[image_id]['instances'] = instances
        self.data[image_id]['file_name'] = file_parts[-1]
        self.data[image_id]['id'] = image_id
        self.data[image_id]['locations'] = locations


    def get_patch(self, idx, image_id):
        patches = self.sampler(self.data[image_id])
        if len(patches) != self.num_per_image:
            raise RuntimeWarning(
                f"`patch_func` must return a sequence of length: samples_per_image={self.num_per_image}."
            )
        patch_id = (idx - image_id * self.num_per_image) * (-1 if idx < 0 else 1)
        patch = patches[patch_id]


        # Transform array
        # image = patch['image'][0]
        # label = patch['label'][0]
        # # # image = Image.fromarray(patch['image'][0]) # Pillow
        # # # label = Image.fromarray(patch['label'][0])
        # if self.transform_patch is not None: 
        #     image = self.transform_patch(image)
        #     label = self.transform_patch(label) 
        #     # label = torch.tensor(label)
        # patch['image'] = image
        # patch['label'] = label

        # Transform dictionary
        if self.transform_patch is not None: 
            patch = self.transform_patch(patch)

        patch['file_name'] = self.pre_im_file
        patch['id'] = patch_id


        label = patch['label'][0]
        label_2 = seg_widen_border(label, self.LABEL_EROSION) # eroison
        label_3 = seg_to_instance_bd(label, self.boundary_size, self.do_background).astype(np.float32) # boundary
 
        weight_region = seg_to_weights([label_2], wopts=['1']) #  0 (no weight), 1 (weight_binary_ratio), 2 (weight_unet3d)
        weight_boundary = seg_to_weights([label_3], wopts=['1'])
        # labels = np.stack([label, label_3])
        labels = np.stack([label_2, label_3])
        weights = np.stack([weight_region[0][0], weight_boundary[0][0]])
        patch['labels'] = labels
        patch['weights'] = weights
        
        return patch

    
    def get_patch_2(self, idx, image_id):
        # patches = self.sampler(self.data[image_id])
        # if len(patches) != self.num_per_image:
        #     raise RuntimeWarning(
        #         f"`patch_func` must return a sequence of length: samples_per_image={self.num_per_image}."
        #     )
        # patch_id = (idx - image_id * self.num_per_image) * (-1 if idx < 0 else 1)
        # patch = patches[patch_id]

        patch = self.crop_patch(image_id)


        # Transform array
        # image = patch['image'][0]
        # label = patch['label'][0]
        # # # image = Image.fromarray(patch['image'][0]) # Pillow
        # # # label = Image.fromarray(patch['label'][0])
        # if self.transform_patch is not None: 
        #     image = self.transform_patch(image)
        #     label = self.transform_patch(label) 
        #     # label = torch.tensor(label)
        # patch['image'] = image
        # patch['label'] = label

        # Transform dictionary
        if self.transform_patch is not None: 
            # print(patch['image'].shape)
            # patch['image'] = patch['image'][0].transpose(2, 0, 1)
            # patch['label'] = patch['label'][0].transpose(2, 0, 1)
            patch = self.transform_patch(patch)
            # patch['image'] = patch['image'].transpose(1, 2, 0)
            # patch['label'] = patch['label'].transpose(1, 2, 0)
            # patch['image'] = np.expand_dims(patch['image'], axis=0)
            # patch['label'] = np.expand_dims(patch['label'], axis=0)
            # patch['image'] = Ft.to_tensor(patch['image'])
            # patch['label'] = Ft.to_tensor(patch['label'])
            # patch['image'] = torch.tensor(patch['image'])
            # patch['label'] = torch.tensor(patch['label'])
            # patch['image'] = patch['image'].unsqueeze(dim=0)
            # patch['label'] = patch['label'].unsqueeze(dim=0)

            # If convert to torch.tensor
            # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
            # numpy array has undergone such operation: image = image[..., ::-1]
            # A simple fix is to do: image = image[..., ::-1] - np.zeros_like(image)
            # Or
            # ndarray.copy() will alocate new memory for numpy array which make it normal, I mean the stride is not negative any more.
            # np_array() = torch.tensor(np_array.copy())

            # patch['image'] = np.expand_dims(patch['image'].copy(), axis=0)
            # patch['label'] = np.expand_dims(patch['label'].copy(), axis=0)

            # Torch dataloader will auto convert to torch.tensor if array is numpy.array


        patch['file_name'] = self.pre_im_file
        patch['id'] = idx

        label = patch['label'][0]
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        label_2 = seg_widen_border(label, self.LABEL_EROSION) # eroison
        label_3 = seg_to_instance_bd(label_2, self.boundary_size, self.do_background).astype(np.float32) # boundary, for seeding, boundary should be built from the label_2 (eroison)
        
        weight_region = seg_to_weights([label_2], wopts=['1']) #  0 (no weight), 1 (weight_binary_ratio), 2 (weight_unet3d)
        # weight_boundary = seg_to_weights([label_3], wopts=['1'])
        weight_boundary = seg_to_weights([patch['shell'][0]], wopts=['1'])

        # labels = np.stack([label, label_3])
        # labels = np.stack([label_2, label_3])
        labels = np.stack([label_2, patch['shell'][0]])
        weights = np.stack([weight_region[0][0], weight_boundary[0][0]])

        patch['labels'] = labels
        patch['weights'] = weights

        return patch

    def get_patch_center(self, image_dims, patch_size, center_list):
        # center = center_list[np.random.choice(len(center_list), size=1, p=center_list[:,3])][0]
        center = center_list[np.random.choice(len(center_list), size=1)][0]

        shift_range = np.round(0.25 * patch_size).astype(np.int)

        center = np.array(center).astype(np.int)
        x = center[0]
        y = center[1]
        z = center[2]
            
        # Add random shift to coordinates for augmentation:
        x = x + np.random.choice(range(-shift_range[0], shift_range[0]+1))
        y = y + np.random.choice(range(-shift_range[1], shift_range[1]+1))
        z = z + np.random.choice(range(-shift_range[2], shift_range[2]+1))
        
        # Move inside if the center is too close to border:
        if (x<patch_size[0]/2) : x = np.ceil(patch_size[0]/2)
        if (y<patch_size[1]/2) : y = np.ceil(patch_size[1]/2)
        if (z<patch_size[2]/2) : z = np.ceil(patch_size[2]/2)
        if (x>image_dims[0]-patch_size[0]/2): x = image_dims[0] - np.ceil(patch_size[0]/2)
        if (y>image_dims[1]-patch_size[1]/2): y = image_dims[1] - np.ceil(patch_size[1]/2)
        if (z>image_dims[2]-patch_size[2]/2): z = image_dims[2] - np.ceil(patch_size[2]/2)

        
        return np.array([x, y, z]).astype(np.int)

    def crop_patch(self, image_id):
        center = self.get_patch_center(self.data[image_id]['image'].shape[1:], self.patch_size, self.data[image_id]['locations'])


        half = self.patch_size // 2

        # calculate subtomogram corners
        x = center[0] - half[0], center[0] + half[0]
        y = center[1] - half[1], center[1] + half[1]
        z = center[2] - half[2], center[2] + half[2]


        patch = {}
        # load reconstruction and ground truths
        patch['image'] = self.data[image_id]['image'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        patch['label'] = self.data[image_id]['label'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        patch['shell'] = self.data[image_id]['shell'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        patch['instances'] = self.data[image_id]['instances'][:, x[0]:x[1], y[0]:y[1], z[0]:z[1]]


        return patch
      
                
                    
    def __len__(self):
        """
        return length of the dataset
        """
        if self.num_per_image > 0:
            return self.num_per_image * len(self.data_list)
        else:
            return len(self.data_list)
        
    
    def __getitem__(self, idx):
        """
        For given index, return images with resize and preprocessing.
        """
        if self.num_per_image > 0:
            # READ FULL IMAGE AND LABEL
            image_id = int(idx/self.num_per_image)

            # if image_id != self.pre_im_id:
            #     self.read_full_image(image_id)

            if len(self.data) < image_id+1:
                self.read_full_image(image_id)

            # CROP PATCHES FROM DATA DICTIONARY
            if self.set_type==0 or self.set_type==1: # train or validation
                patch = self.get_patch_2(idx, image_id) # crop by random center
            else:
                patch = self.get_patch(idx, image_id) # random crop

            return patch

        else:
            self.read_full_image(idx)
            return self.data[idx]

                
# endregion DATASET CLASS


# %%
# %%
# region DATASET FUNCTION #################################
###########################################################

def get_data(data_root='data', ids=None, batch_size=(2,2,2,2), num_workers=0, num_patches=(2200,1700,0,0), patch_size=((144, 144, 144), (96, 96, 96), (160, 160, 160)), cfg=None):



    # %%
    ###### DATASETS ######
    #########################################################################################


    # %%
    # ------ DATA PATH ------
    # --------------------------------------------------------------------------------------

    data_train_path = data_root + '/rat/images'
    data_val_path = data_root + '/rat/images'
    data_test_path = data_root + '/rat/images'


    train_list = glob.glob(data_train_path + '/*.mat')
    val_list = glob.glob(data_val_path + '/*.mat')
    test_list = glob.glob(data_test_path + '/*.mat')

    if ids is not None:
        train_list = map(train_list.__getitem__, ids[0])
        val_list = map(val_list.__getitem__, ids[1])
        test_list = map(test_list.__getitem__, ids[2])
        train_list = list(train_list)
        val_list = list(val_list)
        test_list = list(test_list)
        

    im_resize = 1024


    patch_size_train = patch_size[0]
    patch_size_val = patch_size[1]
    patch_size_test = patch_size[2]


    
    # %%
    # data_train_path

    # %%
    # ------ DATASET (PLAIN) ------
    # --------------------------------------------------------------------------------------
    # train_dataset =  PatchDataset(data_train_path, train_list, set_type=0, n_patches=num_patches[0], patch_size=patch_size_train, 
    #                                 image_shape=None, transform=None, transform_patch=None)
    # print('Length of train dataset: {}'.format(len(train_dataset)))

    # val_dataset =  PatchDataset(data_val_path, val_list, set_type=1, n_patches=num_patches[1], patch_size=patch_size_val, 
    #                                 image_shape=None, transform=None, transform_patch=None)
    # print('Length of valid dataset: {}'.format(len(val_dataset)))

    # eval_dataset =  PatchDataset(data_val_path, val_list, set_type=1, n_patches=num_patches[2], patch_size=patch_size_val, 
    #                                 image_shape=None, transform=None, transform_patch=None)
    # print('Length of evalation dataset: {}'.format(len(eval_dataset)))

    # test_dataset =  PatchDataset(data_test_path, test_list, set_type=2, n_patches=num_patches[3], patch_size=patch_size_test, 
    #                                 image_shape=None, transform=None, transform_patch=None)
    # print('Length of test dataset: {}'.format(len(test_dataset)))

    # %%
    # val_dataset.data

    # %%
    # ------ EXAMPLE (PLAIN - TRAIN) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex1 = train_dataset[i]
    # # ex1 = train_dataset.__getitem__(i)
    # img, trgt = ex1['image'], ex1['label']
    # print('\nEXAMPLE (PLAIN - TRAIN)')
    # # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(np.array(img))
    # plt.title('Train Image ' + str(i) + ': ' +  ex1['file_name'])
    # plt.subplot(122); plt.imshow(np.array(trgt))
    # plt.title('Label')
    # plt.show()

    # %%
    # # train_data = train_dataset.__getitem__(4)
    # train_data = train_dataset[5]
    # image, label = (train_data['image'][0], train_data['label'][0]) # 3D images
    # print('Train data')
    # # print(train_data['im_file'])
    # print('\nimage shape: {}, label shape: {}\n'.format(image.shape, label.shape))

    # # %%
    # plot_volume_interactive(image)

    # # %%
    # plot_volume_interactive(label)



   
    # %%
    # # val_data = val_dataset.__getitem__(4)
    # val_data = val_dataset[700]
    # image, label = (val_data['image'][0], val_data['label'][0]) # 3D images
    # # print(val_data['im_file'])
    # print('\nimage shape: {}, label shape: {}\n'.format(image.shape, label.shape))

    # # %%
    # plot_volume_interactive(image)

    # # %%
    # plot_volume_interactive(label)

    # # %%
    # # eroison
    # label_2 = val_data['labels'][0]
    # plot_volume_interactive(label_2)

    # # %%
    # # boundary
    # label_3 = val_data['labels'][1]
    # plot_volume_interactive(label_3)



    







    # %%
    # ------ EXAMPLE (PLAIN - VALID) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex2 = val_dataset[i]
    # # ex1 = train_dataset.__getitem__(i)
    # img, trgt = ex2['image'], ex2['label']
    # print('\nEXAMPLE (PLAIN - VALID)')
    # # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(np.array(img))
    # plt.title('Valid Image ' + str(i) + ': ' +  ex2['file_name'])
    # plt.subplot(122); plt.imshow(np.array(trgt))
    # plt.title('Label')
    # plt.show()


    # %%
    # ------ EXAMPLE (PLAIN - EVALUATION) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex2 = eval_dataset[i]
    # # ex1 = train_dataset.__getitem__(i)
    # img, trgt = ex2['image'], ex2['label']
    # print('\nEXAMPLE (PLAIN - EVALUATION)')
    # # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(np.array(img))
    # plt.title('Valid Image ' + str(i) + ': ' +  ex2['file_name'])
    # plt.subplot(122); plt.imshow(np.array(trgt))
    # plt.title('Label')
    # plt.show()

    # %%
    # ------ EXAMPLE (PLAIN - TEST) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex3 = test_dataset[i]
    # img3 = ex3['image']
    # print('\nEXAMPLE (PLAIN - TEST)')
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.imshow(np.array(img3))
    # plt.show()





    # %%
    ###### TRANSFORMATION ######
    #########################################################################################

    

    # %%
    def image_preprocess_transforms():
        
        preprocess = transforms.Compose([
            # transforms.Resize(512),
            transforms.Resize(im_resize),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor()
            ])
        
        return preprocess



    # %%
    def get_mean_std(data_root, batch_size=8, num_workers=0):
        
        pre_transforms = transforms.Compose([
            # image_preprocess_transforms(),
            # By default the imageFolder loads images with 3 channels and we expect the image to be grayscale.
            # So let's transform the image to grayscale
            # transforms.Grayscale(),
            transforms.ToTensor(), # transpose from H*W*C to C*H*W
        ])

         
        
        class MyDataset(Dataset):
            """custom dataset for .mat images"""

            def __init__(self, list_of_urls, transforms):
                self.list_of_urls = list_of_urls
                self.transform = transforms

            def __len__(self):
                return len(self.list_of_urls)

            def __getitem__(self, index):
                image_url = self.list_of_urls[index]
                image = hdf5storage.loadmat(image_url)
                image = pre_transforms(image['outImg'])
                # print(image)
                return image
        
        if data_root is not None:
            dataset = datasets.ImageFolder(root=data_root, transform=pre_transforms)
        else:
            dataset = MyDataset(test_list, pre_transforms)
        
        # print(len(dataset))
        # image = dataset[1]
        # print(type(image[0]))
        # print(image[0].size)

        # n = len(dataset)
        # n_sample = 3200
        # sample_ids = np.random.choice(range(n), n_sample, replace=False) # replace=False: unrepeative

        
        
        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            # sampler=SubsetRandomSampler(sample_ids),
                                            shuffle=False)

        mean = 0.
        std = 0.

        
    
        for images in loader:
            # print(images)
            # print(type(images[0]))
            batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(loader.dataset)
        std /= len(loader.dataset)
        
        print('mean: {}, std: {}'.format(mean, std))
        
        return mean, std




    # %%
    def image_common_transforms(mean=(0.5671, 0.4666, 0.3664), std=(0.2469, 0.2544, 0.2584)):
        preprocess = image_preprocess_transforms()
        
        common_transforms = transforms.Compose([
            # preprocess,
            transforms.ToTensor(), # transpose from H*W*C to C*H*W
            transforms.Normalize(mean, std)
        ])
        
        return common_transforms


    # %%
    def data_augmentation_preprocess(mean=(0.5671, 0.4666, 0.3664), std=(0.2469, 0.2544, 0.2584)):
    
        preprocess = image_preprocess_transforms()

        augmentation_transforms = transforms.Compose([
            # preprocess,
            # transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(round(0.875*im_resize)),
            transforms.RandomAffine(degrees=30),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(), # transpose from H*W*C to C*H*W
            transforms.Normalize(mean, std)
                                                    ])
        return augmentation_transforms


    # %%
    def patch_transform():
        tf = transforms.ToTensor()
        return tf


    def monai_transforms():
        # Define transforms for image
        train_transforms = Compose(
            # [   
            #     # AddChanneld(keys=["image", "label", 'shell']),
            #     # Resized(keys=["image", "label"], spatial_size=(512, 512)),
            #     # RandRotated(keys=["image", "label"], prob=0.2, range_x=(30/180), range_y=(30/180), range_z=(30/180), padding_mode='border'),
            #     # RandAffined(keys=["image", "label"], prob=0.2, translate_range=(10,10,10), padding_mode='border'),
            #     RandAffined(keys=["image", "label"], prob=0.001, translate_range=(1,1,1), padding_mode='border'),
            #     # RandZoomd(keys=["image", "label"], prob=0.30, min_zoom=0.90, max_zoom=1.1, padding_mode='edge', keep_size=True),
            #     ToTensord(keys=["image", "label"]), # data loader will automatically convert to tensor
            # ]
            [   
                # AddChanneld(keys=["image", "label"]),
                # RandRotated(keys=["image", "label"], prob=0.5, range_x=(90/180), range_y=(90/180), range_z=(90/180), padding_mode='border'),
                # RandRotate90d(keys=['image', 'label', 'shell'], prob=0.7, spatial_axes=(1,2)), 
                RandRotate90d(keys=['image', 'label', 'shell', 'instances'], prob=0.7, spatial_axes=(0,1)), # count 0, ignore the first index of batch
                # RandAxisFlipd(keys=['image', 'label'], prob=0.5),
                # RandAffined(keys=["image", "label"], prob=0.2, rotate_range=(90/180,90/180,90/180), padding_mode='zeros'),
                RandZoomd(keys=["image", "label", 'shell', 'instances'], prob=0.25, min_zoom=0.8, max_zoom=1.25, padding_mode='constant', keep_size=True),
                # Rand3DElasticd(keys=["image", "label"], prob=0.1, sigma_range=(3,4), magnitude_range=(2,2), padding_mode='zeros'),
                RandAdjustContrastd(keys=['image'], gamma=1.2, prob=0.2),
                # RandAdjustContrastd(keys=['image'], gamma=1.1, prob=0.25),
                # RandShiftIntensityd(keys=['image'], offsets=[0.2,0.8], prob=0.1),
                RandGaussianNoised(keys=['image'], prob=0.2, mean=0.3, std=0.1),
                RandGaussianSmoothd(keys=['image'], prob=0.2),
                # ToTensord(keys=["image", "label"]), # data loader will automatically convert to tensor
            ],
            # [   
            #     # AddChanneld(keys=["image", "label"]),
            #     # RandRotated(keys=["image", "label"], prob=0.5, range_x=(90/180), range_y=(90/180), range_z=(90/180), padding_mode='border'),
            #     RandAffined(keys=["image", "label", 'instances'], prob=0.25, rotate_range=(90/180,90/180,90/180), padding_mode='constant'),
            #     RandZoomd(keys=["image", "label", 'instances'], prob=0.25, min_zoom=0.8, max_zoom=1.25, padding_mode='constant', keep_size=True),
            #     Rand3DElastic(key=["image", "label", 'instances'], prob=0.1, sigma_range=(3,4),  padding_mode='constant'),
            #     RandGaussianNoise(key=['image'], prob=0.25, mean=0.3, std=0.1),
            #     RandGaussianSmooth(key=['image'], prob=0.25),
            #     ToTensord(keys=["image", "label"]), # data loader will automatically convert to tensor
            # ]
            
        )
        val_transforms = train_transforms

        eval_transforms = Compose(
            [
                # AddChanneld(keys=["image", "label"]),
                # Resized(keys=["image", "label"], spatial_size=(512, 512)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = eval_transforms

        return train_transforms, val_transforms, eval_transforms, test_transforms




    # %%
    ###### DATA LOADER ######
    #########################################################################################

    # %%
    # ------ NORMALIZE ------
    # --------------------------------------------------------------------------------------
    # data_root_2 = os.path.join(data_root, 'training/images') # NOT WORKING

    # batch_size = 6# DEBUG
    # num_workers = 0 # DEBUG

    # batch_size_train = batch_size[0]
    # batch_size_test = batch_size[0]

        
    # mean, std = get_mean_std(data_root=None, batch_size=batch_size, num_workers=num_workers)
    mean = 0.5008
    std = 0.2637

  

    # %%
    # ------ TRANSFORM ------
    # --------------------------------------------------------------------------------------

    # Image transforms
    common_transforms = image_common_transforms(mean, std)
    data_augmentation = False
    if data_augmentation:    
        train_transforms = data_augmentation_preprocess(mean, std)
    else:
        train_transforms = common_transforms

    # Patch transforms
    train_transforms, val_transforms, eval_transforms, test_transforms = monai_transforms()
    check_transforms = Compose(
                    Resize(spatial_size=(512,512,100))
    )

    # Connectomics transform
    # train_transforms = build_train_augmentor(cfg)
    # test_transforms = TestAugmentor(mode = cfg.INFERENCE.AUG_MODE, 
    #                                 do_2d = cfg.DATASET.DO_2D,
    #                                 num_aug = cfg.INFERENCE.AUG_NUM,
    #                                 scale_factors = cfg.INFERENCE.OUTPUT_SCALE)
    # if not cfg.DATASET.DO_CHUNK_TITLE:
    #     test_filename = cfg.INFERENCE.OUTPUT_NAME
    #     test_filename =test_transforms.update_name(test_filename)

        



    # %%
    # ------ RELOAD DATASETS (TRANSFORMS) ------
    # --------------------------------------------------------------------------------------

    # train_dataset =  PatchDataset(data_root, train_list, set_type=0, image_shape=im_resize, transform=train_transforms)
    # print('Length of train dataset: {}'.format(len(train_dataset)))

    # val_dataset =  PatchDataset(data_root, val_list, set_type=1, image_shape=im_resize, transform=common_transforms)
    # print('Length of valid dataset: {}'.format(len(val_dataset)))


    train_dataset =  PatchDataset(data_root, train_list, set_type=0, n_patches=num_patches[0], patch_size=patch_size_train, 
                                    image_shape=None, transform=None, transform_patch=train_transforms)
    # print('Length of train dataset: {}'.format(len(train_dataset)))

    val_dataset =  PatchDataset(data_root, val_list, set_type=1, n_patches=num_patches[1], patch_size=patch_size_val, 
                                    image_shape=None, transform=None, transform_patch=None)
    # print('Length of valid dataset: {}'.format(len(val_dataset)))

    eval_dataset =  PatchDataset(data_root, val_list, set_type=1, n_patches=num_patches[2], patch_size=patch_size_val, 
                                    image_shape=None, transform=None, transform_patch=None)
    print('Length of evaluation dataset: {}'.format(len(eval_dataset)))

    test_dataset =  PatchDataset(data_root, test_list, set_type=2, n_patches=num_patches[3], patch_size=patch_size_test, 
                                    image_shape=None, transform=None, transform_patch=None)
    # print('Length of test dataset: {}'.format(len(test_dataset)))

    # %%
    # ------ EXAMPLE (TRANSFORMS - TRAIN) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex1 = train_dataset[i]
    # # ex1 = train_dataset.__getitem__(i)
    # img, trgt = ex1['image'], ex1['label']
    # print('\nEXAMPLE (TRANSFORMS - TRAIN)')
    # # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(1-np.array(img.permute(1,2,0)))
    # plt.title('Train Image ' + str(i) + ': ' +  ex1['file_name'])
    # plt.subplot(122); plt.imshow(1-np.array(trgt.permute(1,2,0)))
    # plt.title('Label')
    # plt.show()

    # # %%
    # torch.max(img)

    # # %%
    # torch.min(img)

    # # %%
    # np.unique(np.array(trgt))

 


    # %%
    # ------ EXAMPLE (TRANSFORMS - VALID) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex2 = val_dataset[i]
    # # ex1 = train_dataset.__getitem__(i)
    # img2, trgt2 = ex2['image'], ex2['label']
    # print('\nEXAMPLE (TRANSFORMS - VALID)')
    # # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(1-np.array(img2.permute(1,2,0)))
    # plt.title('Train Image ' + str(i) + ': ' +  ex2['file_name'])
    # plt.subplot(122); plt.imshow(1-np.array(trgt2.permute(1,2,0)))
    # plt.title('Label')
    # plt.show()



    # %%
    # # CHOICE
    # # val_data = val_ds.__getitem__(4)
    # # val_data = val_dataset[5]
    # val_data = train_dataset[5]
    # image, label = (val_data['image'][0], val_data['label'][0]) # 3D images
    # print('Validation data')
    # # print(val_data['im_file'])
    # print('\nimage shape: {}, label shape: {}\n'.format(image.shape, label.shape))

    # # %%
    # plot_volume_interactive(image)

    # # %%
    # plot_volume_interactive(label)

    # # %%
    # labels = val_data['labels']

    # print(type(labels))
    # print(labels.shape)
    # # <class 'numpy.ndarray'>
    # # (2, 256, 256, 64)


    # # %%
    # plot_volume_interactive(labels[0])

    # # %%
    # plot_volume_interactive(labels[1])


    # %%
    # image_2 = torch.tensor(image)



    # %%
    # ------ EXAMPLE (TRANSFORMS - EVALUATION) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex2b = eval_dataset[i]
    # # ex1 = train_dataset.__getitem__(i)
    # img2b, trgt2b = ex2b['image'], ex2b['label']
    # print('\nEXAMPLE (TRANSFORMS - EVALUATION)')
    # # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(1-np.array(img2b.permute(1,2,0)))
    # plt.title('Train Image ' + str(i) + ': ' +  ex2b['file_name'])
    # plt.subplot(122); plt.imshow(1-np.array(trgt2b.permute(1,2,0)))
    # plt.title('Label')
    # plt.show()


    # %%
    # ------ EXAMPLE (TRANSFORMS - TEST) ------
    # --------------------------------------------------------------------------------------
    # i = 1
    # ex3 = test_dataset[i]
    # img3 = ex3['image']
    # print('\nEXAMPLE (TRANSFORMS - TEST)')   
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.imshow(1-np.array(img3.permute(1,2,0)), vmin=torch.min(img3), vmax=torch.max(img3))
    # plt.title('Test Image ' + str(i) + ': ' +  ex3['file_name'])
    # plt.show()


    # %%
    # torch.max(img3)

    # %%
    # torch.min(img3)



    # %%
    # ------ LOADERS ------
    # --------------------------------------------------------------------------------------
    # num_workers = 0

    train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size[0], 
                                shuffle=True,
                                # sampler=train_sampler, 
                                num_workers=num_workers)


    val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size[1], 
                                shuffle=False, 
                                num_workers=num_workers)

    eval_loader = DataLoader(eval_dataset, 
                                batch_size=batch_size[2], 
                                shuffle=False, 
                                num_workers=num_workers)


    test_loader = DataLoader(test_dataset, 
                                batch_size=batch_size[3], 
                                shuffle=False, 
                                num_workers=num_workers)

    # %%
    # ------ EXAMPLE (LOADER - TRAIN) ------
    # --------------------------------------------------------------------------------------
    # iterloader = iter(train_loader)
    # ex = next(iterloader)
    # img, trgt, file_name = ex['image'], ex['label'], ex['file_name']
    # print('\nEXAMPLE (LOADER - TRAIN)')  
    # # plt.imshow(img[0].permute(1,2,0))
    # plt.figure(figsize=(10, 8))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(np.array(img[0].permute(1,2,0)))
    # plt.title('Validation Image: ' +  file_name[0])
    # plt.subplot(122); plt.imshow(np.array(trgt[0].permute(1,2,0)))
    # plt.title('Label')
    # plt.show()


    # %%
    # ------ EXAMPLE (LOADER - VALID) ------
    # --------------------------------------------------------------------------------------
    # iterloader = iter(val_loader)
    # ex = next(iterloader)
    # img, trgt, file_name = ex['image'], ex['label'], ex['file_name']
    # print('\nEXAMPLE (LOADER - VALID)') 
    # # plt.imshow(img[0].permute(1,2,0))
    # plt.figure(figsize=(10, 8))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(np.array(img[0].permute(1,2,0)))
    # plt.title('Validation Image: ' +  file_name[0])
    # plt.subplot(122); plt.imshow(np.array(trgt[0].permute(1,2,0)))
    # plt.title('Label')
    # plt.show()

    # %%
    # # CHOICE
    # # iterloader = iter(val_loader)
    # iterloader = iter(train_loader)
    # val_data = next(iterloader)
    # image, label = (val_data['image'], val_data['label']) # 3D images
    # print('Validation data')
    # # print(val_data['im_file'])
    # print('\nimage shape: {}, label shape: {}\n'.format(image.shape, label.shape))

    # # %%
    # plot_volume_interactive(image[0,0].numpy())

    # # %%
    # plot_volume_interactive(label[0,0].numpy())

    # # %%
    # labels = val_data['labels']

    # print(type(labels))
    # print(labels.shape)



    # # %%
    # plot_volume_interactive(labels[0,0].numpy())

    # # %%
    # plot_volume_interactive(labels[0,1].numpy())



    # %%
    # ------ EXAMPLE (LOADER - EVALUATION) ------
    # --------------------------------------------------------------------------------------
    # iterloader = iter(eval_loader)
    # ex = next(iterloader)
    # img, trgt, file_name = ex['image'], ex['label'], ex['file_name']
    # print('\nEXAMPLE (LOADER - EVALUATION)') 
    # # plt.imshow(img[0].permute(1,2,0))
    # plt.figure(figsize=(10, 8))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.subplot(121); plt.imshow(np.array(img[0].permute(1,2,0)))
    # plt.title('Validation Image: ' +  file_name[0])
    # plt.subplot(122); plt.imshow(np.array(trgt[0].permute(1,2,0)))
    # plt.title('Label')
    # plt.show()

    # %%
    # ------ EXAMPLE (LOADER - TEST) ------
    # --------------------------------------------------------------------------------------
    # iterloader = iter(test_loader)
    # ex = next(iterloader)
    # img, file_name = ex['image'], ex['file_name']
    # print('\nEXAMPLE (LOADER - TEST)') 
    # # plt.imshow(img[0].permute(1,2,0))
    # plt.figure(figsize=(6, 4))
    # plt.rcParams.update({'font.size': 6.5})
    # plt.imshow(np.array(img[0].permute(1,2,0)))
  
  


    # %%

    pdata = {}
    pdata['train_dataset'] = train_dataset
    pdata['val_dataset'] = val_dataset
    pdata['eval_dataset'] = eval_dataset
    pdata['test_dataset'] = test_dataset

    pdata['train_loader'] = train_loader
    pdata['val_loader'] = val_loader
    pdata['eval_loader'] = eval_loader
    pdata['test_loader'] = test_loader

    
    return pdata