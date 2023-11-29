import os
import torch
from skimage.measure.simple_metrics import compare_psnr
from tqdm import tqdm
import glob
import numpy as np
from PIL import Image
import math

def pil_loader(path):
    img = Image.open(path)
    if len(img.getbands()) == 1:
        print("gray img in :",path)
        return img.convert("RGB")
    return img.convert("RGB")


def do_psnr(img, imgclean, data_range=255):

	img = np.array(img).astype(np.float32)
	imgclean = np.array(imgclean).astype(np.float32)
	#print(img.shape , imgclean.shape)
	psnr = compare_psnr(imgclean, img, \
					data_range=data_range)
	return psnr


def do_psnr2(img1, img2):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2


def calculate_psnr(target_imgs,gt_imgs):
    psnr = 0
    target_imgs = sorted(glob.glob(os.path.join(target_imgs, '*.JPEG'))+glob.glob(os.path.join(target_imgs, '*.jpg'))+glob.glob(os.path.join(target_imgs, '*.png')))
    gt_imgs = sorted(glob.glob(os.path.join(gt_imgs, '*.JPEG'))+glob.glob(os.path.join(gt_imgs, '*.jpg'))+glob.glob(os.path.join(gt_imgs, '*.png')))
    n = len(target_imgs)
    for i in tqdm(range(n)):
        target_img = pil_loader(target_imgs[i])
        gt_img = pil_loader(gt_imgs[i])
        psnr+=do_psnr2(target_img,gt_img)
    return psnr/n




if __name__ == "__main__":
    target_dir = "/dataset/checkpoints/spcolor/tps"
    gt_dir = "/dataset/tps/imgs"
    psnr = calculate_psnr(target_dir,gt_dir)
    print("--------psnr--------:",psnr)