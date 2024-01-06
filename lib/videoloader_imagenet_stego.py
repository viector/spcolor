import os
import os.path as osp
import struct

import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from utils.util_distortion import CenterPadCrop_numpy,CenterPad, Distortion_with_flow, Normalize, RGB2Lab, ToTensor
import yaml
from traceback import print_exc

cv2.setNumThreads(0)

def get_yaml(dir,name):
    dir_grays = osp.join(dir,name)
    f_gray=open(dir_grays,'r',encoding='utf-8')
    gray_read=f_gray.read()
    gray_dict=yaml.load(gray_read, Loader=yaml.FullLoader) 
    #print("file in get:",gray_dict)
    f_gray.close()
    return gray_dict

def parse_images(dir):
    print("dir is: ", dir)
    dir = osp.expanduser(dir)
    dir_analogy = osp.join(dir,"pairs/analogies")
    gray_dict = get_yaml(dir,"gray_imgs.yaml")
    small_dict = get_yaml(dir,"small_imgs.yaml")
    error_dict = get_yaml(dir,"error_imgs.yaml")
    mono_dict = get_yaml(dir,"mono_imgs.yaml")
    image_pairs = []
    bad_num = 0
    analogies = sorted(glob.glob(os.path.join(dir_analogy, '*.npy')))
    for i in range(len(analogies)):
        analogy_path = analogies[i]
        analogy = np.load(analogy_path,mmap_mode = 'r')
        analogy_path = analogy_path.split("analogies/")[1]
        #print("yamls:",mono_dict,"analogy_path",analogy_path)
        bad_list = list(set(gray_dict[analogy_path]+small_dict[analogy_path]+mono_dict[analogy_path]+error_dict[analogy_path]))
        for j in range(len(analogy)):
            pair = analogy[j]
            if pair[5] in bad_list:
                bad_num+=1
                continue
            for r in range(5):
                if pair[r] in bad_list:
                    bad_num+=1
                    continue
                item0 =  (dir, analogy_path, pair[r], pair[5], r)
                #item1 =  (dir, analogy_path, pair[5], pair[r], 2)
                image_pairs.append(item0)
                #image_pairs.append(item1)
                    # if float(pair[2]) > 0:
                    #     item0 = (dir, target, name0, name1, 2)
                    #     item1 = (dir, target, name1, name0, 2)
                    #     image_pairs.append(item0)
                    #     image_pairs.append(item1)
    #f_gray.close()
    print("total imgs:",len(image_pairs),"total bad imgs:",bad_num)
    return image_pairs


def pil_loader(path):
    img = Image.open(path)
    #if img.layers == 1:
    if len(img.getbands()) == 1:
        print("gray img in :",path)
        return img.convert("RGB"),0
    return img.convert("RGB"),1

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class VideosDataset_ImageNet(data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        with_bad=False,
        with_mid=False,
        transforms_imagenet=None,
        distortion_level=3,
        brightnessjitter=0,
        nonzero_placeholder_probability=0.5,
        extra_reference_transform=None,
        real_reference_probability=1,
    ):
        image_pairs = []
        curr_image_pairs = parse_images(data_root)
        image_pairs += curr_image_pairs
        print("##### parsing image_a pairs in %s: %d pairs #####" % (data_root, len(curr_image_pairs)))
        if not image_pairs:
            raise RuntimeError("Found 0 image_a pairs in all the data_roots")

        self.image_pairs = image_pairs
        self.transforms_imagenet_raw = transforms_imagenet
        self.extra_reference_transform = transforms.Compose(extra_reference_transform)
        self.real_reference_probability = real_reference_probability
        self.distortion_ref_prob = (1-real_reference_probability)/2
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        #self.epoch = epoch
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        #self.image_pairs *= epoch
        self.distortion_level = distortion_level
        self.distortion_transform = Distortion_with_flow()
        self.brightnessjitter = brightnessjitter
        self.flow_transform = transforms.Compose([CenterPadCrop_numpy(self.image_size), ToTensor()])
        self.nonzero_placeholder_probability = nonzero_placeholder_probability
        self.ToTensor = ToTensor()
        self.Normalize = Normalize()
        self.transforms_stego = transforms.Compose([CenterPad(image_size),transforms.ToTensor(),normalize])
        

    def distortion_ref(self,I2):
        ## generate the flow
            height, width = np.array(I2).shape[0], np.array(I2).shape[1]
            alpha = np.random.rand() * self.distortion_level
            distortion_range = 50
            random_state = np.random.RandomState(None)
            shape = self.image_size[0], self.image_size[1]
            # dx: flow on the vertical direction; dy: flow on the horizontal direction
            forward_dx = (
                gaussian_filter((random_state.rand(*shape) * 2 - 1), distortion_range, mode="constant", cval=0)
                * alpha
                * 1000
            )
            forward_dy = (
                gaussian_filter((random_state.rand(*shape) * 2 - 1), distortion_range, mode="constant", cval=0)
                * alpha
                * 1000
            )

            for transform in self.transforms_imagenet_raw:
                if type(transform) is RGB2Lab:
                    I2 = self.distortion_transform(I2, forward_dx, forward_dy)
                    I2_raw = I2
                I2 = transform(I2)
            I2[0:1, :, :] = I2[0:1, :, :] + torch.randn(1) * self.brightnessjitter
            return I2


    def __getitem__(self, index):
        try:
            #print("get item")
            pair_id = index

            combo_path = None
            image_a_path = None
            image_b_path = None

            dir_root, cls_dir, image_names0, image_names1, is_good = self.image_pairs[pair_id]

            image_a_path = osp.join(dir_root, "imgs", image_names0)
            image_b_path = osp.join(dir_root, "imgs", image_names1)

            # if np.random.random() > 0.5:
            #     image_a_path, image_b_path = image_b_path, image_a_path

            I1,flag = pil_loader(image_a_path)
            if flag == 0:
                print("gray img in:",image_a_path)
                return self.__getitem__(np.random.randint(0, len(self.image_pairs)))

            # got reference
            seed = np.random.random()
            if seed < self.real_reference_probability:
                I_reference_video_real,flag = pil_loader(image_b_path)
                if flag == 0:
                    print("gray img in:",image_b_path)
                    return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
                I_reference_video_lab = self.transforms_imagenet(I_reference_video_real)
                I_reference_output = I_reference_video_lab
                self_ref_flag = torch.zeros_like(I_reference_output)
                I_reference_video_rgb = self.transforms_stego(I_reference_video_real)
            # elif seed < self.real_reference_probability + self.distortion_ref_prob:
            #     I_reference_video = self.distortion_ref(I1)
            #     I_reference_output = I_reference_video
            #     self_ref_flag = torch.ones_like(I_reference_output)
            else:
                I_reference_video = I1
                I_reference_video = self.extra_reference_transform(I_reference_video)
                I_reference_lab = self.transforms_imagenet(I_reference_video)
                I_reference_output = I_reference_lab
                self_ref_flag = torch.ones_like(I_reference_output)
                I_reference_video_rgb = self.transforms_stego(I_reference_video)

            I1_lab= self.transforms_imagenet(I1.convert("L").convert("RGB"))
            I1_rgb= self.transforms_stego(I1)
            
            outputs = [
                I1_lab,
                I_reference_output,
                I1_rgb,
                I_reference_video_rgb,
                self_ref_flag,
            ]

        except Exception as e:
            if combo_path is not None:
                print("problem in ", combo_path)
            print("problem in, ", image_a_path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return self.real_len#len(self.image_pairs)


def parse_images_10k(dir):  #refs in folder
    #print("dir is: ", dir)
    dir = osp.expanduser(dir)
    dir_imgs = osp.join(dir,"imgs")
    dir_refs = osp.join(dir,"refs")
    image_pairs = []
    #ref_order-=1

    imgs = sorted(glob.glob(os.path.join(dir_imgs, '*.JPEG')))
    ref_dirs = sorted(os.listdir(dir_refs))
    for i in range(len(imgs)):
        ref_imgs = sorted(glob.glob(os.path.join(dir_refs,ref_dirs[i], '*.JPEG')))
        item0 =  (osp.basename(imgs[i]), imgs[i],ref_imgs)
        image_pairs.append(item0)
    return image_pairs


def parse_images_wild(dir):   #refs not in folder
    #print("dir is: ", dir)
    dir = osp.expanduser(dir)
    dir_imgs = osp.join(dir,"imgs")
    dir_refs = osp.join(dir,"refs")
    image_pairs = []
    #ref_order-=1

    imgs = sorted(glob.glob(os.path.join(dir_imgs, '*.JPEG'))+glob.glob(os.path.join(dir_imgs, '*.jpg'))+glob.glob(os.path.join(dir_imgs, '*.png')))
    #ref_dirs = sorted(os.listdir(dir_refs))
    ref_imgs = sorted(glob.glob(os.path.join(dir_refs, '*.JPEG'))+glob.glob(os.path.join(dir_refs, '*.jpg'))+glob.glob(os.path.join(dir_refs, '*.png')))
    for i in range(len(imgs)):
        item0 =  (osp.basename(imgs[i]), imgs[i],ref_imgs[i])
        image_pairs.append(item0)
    return image_pairs



class VideosDataset_wild(data.Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        transforms_imagenet=None,
        reference_order=1,
        resize_before_crop = True
    ):
        image_pairs = []
        curr_image_pairs = parse_images_wild(data_root)#[682:683] [1221:1222] [534:535]
        image_pairs += curr_image_pairs
        print("##### parsing image_a pairs in %s: %d pairs #####" % (data_root, len(curr_image_pairs)))
        if not image_pairs:
            raise RuntimeError("Found 0 image_a pairs in all the data_roots")

        self.image_pairs = image_pairs
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        self.ToTensor = ToTensor()
        self.Normalize = Normalize()
        self.transforms_stego = transforms.Compose([CenterPad(image_size),transforms.ToTensor(),normalize])
        self.reference_order = reference_order - 1
        self.resize_before_crop = resize_before_crop
        
    def __getitem__(self, index):
        try:
            name, image_a_path, image_b_path = self.image_pairs[index]

            I1,flag = pil_loader(image_a_path)

            # got reference
            I_reference_video_real,flag = pil_loader(image_b_path)
            if self.resize_before_crop:
                I1 = I1.resize([256,256])
                I_reference_video_real = I_reference_video_real.resize([256,256])
            I_reference_video_rgb = self.transforms_stego(I_reference_video_real)
            I_reference_output = self.transforms_imagenet(I_reference_video_real)
            I1_rgb = self.transforms_stego(I1.convert("L").convert("RGB"))
            I1 = self.transforms_imagenet(I1)
            
            outputs = [
                I1,
                I_reference_output,
                I1_rgb,
                I_reference_video_rgb,
                name,
            ]
        except Exception as e:
            print("problem in, ", image_a_path)
            print_exc()
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return self.real_len



class VideosDataset_ImageNet_10k(data.Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        transforms_imagenet=None,
        reference_order=1,
    ):
        image_pairs = []
        curr_image_pairs = parse_images_10k(data_root)#[682:683] [1221:1222] [534:535]
        image_pairs += curr_image_pairs
        print("##### parsing image_a pairs in %s: %d pairs #####" % (data_root, len(curr_image_pairs)))
        if not image_pairs:
            raise RuntimeError("Found 0 image_a pairs in all the data_roots")

        self.image_pairs = image_pairs
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        self.ToTensor = ToTensor()
        self.Normalize = Normalize()
        self.transforms_stego = transforms.Compose([CenterPad(image_size),transforms.ToTensor(),normalize])
        self.reference_order = reference_order - 1
        
    def __getitem__(self, index):
        try:
            name, image_a_path, image_b_paths = self.image_pairs[index]
            name = name.replace("JPEG","png")

            I1,flag = pil_loader(image_a_path)
            if flag == 0:
                print("gray img in target:",image_a_path)
                return self.__getitem__(np.random.randint(0, len(self.image_pairs)))

            # got reference
            for id,image_b_path in enumerate(image_b_paths):
                I_reference_video_real,flag = pil_loader(image_b_path)
                if flag == 1 and id >= self.reference_order:
                    break
            I_reference_video_rgb = self.transforms_stego(I_reference_video_real)
            I_reference_output = self.transforms_imagenet(I_reference_video_real)
            I1_rgb = self.transforms_stego(I1.convert("L").convert("RGB"))
            I1 = self.transforms_imagenet(I1)
            
            outputs = [
                I1,
                I_reference_output,
                I1_rgb,
                I_reference_video_rgb,
                name,
            ]
        except Exception as e:
            print("problem in, ", image_a_path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return self.real_len


def parse_videoclip(dir_imgs):
    #print("dir is: ", dir)
    #dir = osp.expanduser(dir)
    image_pairs = []
    #ref_order-=1

    imgs = sorted(glob.glob(os.path.join(dir_imgs, '*.jpg')))
    print("###len of imgs:", len(imgs),"dir:",dir_imgs)
    ref_img = imgs[0]
    for i in range(len(imgs)):
        item0 =  (osp.basename(imgs[i]), imgs[i],ref_img)
        image_pairs.append(item0)
    return image_pairs


class VideosDataset_videoclip(data.Dataset):
    def __init__(
        self,
        image_size,
        transforms_imagenet=None,
        data_root = "/media/bupt/dataset/DAVIS/Test/1/imgs/girl-dog"
    ):
        image_pairs = []
        curr_image_pairs = parse_videoclip(data_root)
        image_pairs += curr_image_pairs
        print("##### parsing image_a pairs in %s: %d pairs #####" % (data_root, len(curr_image_pairs)))
        if not image_pairs:
            raise RuntimeError("Found 0 image_a pairs in all the data_roots")

        self.image_pairs = image_pairs
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        self.ToTensor = ToTensor()
        self.Normalize = Normalize()
        self.transforms_stego = transforms.Compose([CenterPad(image_size),transforms.ToTensor(),normalize])
        
    def __getitem__(self, index):
        try:
            name, image_a_path, image_b_paths = self.image_pairs[index]
            name = name.replace("jpg","png")
            #print("success",1)
            I1,flag = pil_loader(image_a_path)
            #I1 = I1.resize(self.image_size)
            #print("success",2)
            # got reference
            I_reference_video_real,flag = pil_loader(image_b_paths)
            #I_reference_video_real=I_reference_video_real.resize(self.image_size)
            #print("success",3)
            I_reference_video_rgb = self.transforms_stego(I_reference_video_real)
            I_reference_output = self.transforms_imagenet(I_reference_video_real)
            I1_rgb = self.transforms_stego(I1.convert("L").convert("RGB"))
            I1 = self.transforms_imagenet(I1)
            outputs = [
                I1,
                I_reference_output,
                I1_rgb,
                I_reference_video_rgb,
                name,
            ]
        except Exception as e:
            print("problem in, ", image_a_path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return self.real_len

