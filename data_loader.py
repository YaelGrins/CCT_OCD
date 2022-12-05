import torch
import torch.nn as nn
import numpy as np
import cct7
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist
from torchvision.datasets import cifar
from torchvision import transforms
import Lenet5
from torchvision.datasets import CIFAR100
from cct7 import cct_7_3x1_32_c100

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from copy import deepcopy
import torchvision.transforms as tt

def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':
        data = np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] =get_minibatches
        batch['chunksize'] =chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [],[]
        for img,tfrom in zip(images,tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
                            [
                            transforms.ToTensor()
                            ])
        train_dataset = mnist.MNIST(
                "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
                "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [],[]
        batch_x, batch_y = next(iter(data_loade))
        print(batch_x.shape, batch_y.shape)
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            test_ds.append(deepcopy(batch))
    else:
        args.datatype == 'cifar100'
        #########model_name = 'cct_7_3x1_32_c100'

        img_size = 32
        num_classes = 100
        img_mean = [0.5071, 0.4867, 0.4408]
        img_std = [0.2675, 0.2565, 0.2761]

        model = cct_7_3x1_32_c100(pretrained=False, progress=False)

        normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

        augmentations = []

        from cct_utils.autoaug import CIFAR10Policy
        augmentations += [
            CIFAR10Policy()
            ]
        augmentations += [
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            *normalize,
        ]

        augmentations = transforms.Compose(augmentations)
        train_dataset = cifar.CIFAR100(
            root='./data/', train=True, download=True, transform=augmentations)

        test_dataset = cifar.CIFAR100(
            root='./data/', train=False, download=False, transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                *normalize,
            ]))

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=4)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        batch_x, batch_y = next(iter(train_loader))
        print(batch_x.shape, batch_y.shape)
        train_ds, test_ds = [], []
        for i, (images, target) in enumerate(train_loader):
            batch = {'input': images, 'output': target}
            train_ds.append(deepcopy(batch))
        for i, (images, target) in enumerate(test_loader):
            batch = {'input': images, 'output': target}
            test_ds.append(deepcopy(batch))

    return train_ds, test_ds, model