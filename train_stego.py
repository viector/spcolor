from __future__ import print_function

import argparse
import math
import os
import queue
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform_lib
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import CenterCrop
import torch.nn.functional as F

import lib.TrainTransforms as transforms
from lib.videoloader import VideosDataset
from lib.videoloader_imagenet_stego import VideosDataset_ImageNet , VideosDataset_ImageNet_10k, VideosDataset_videoclip,VideosDataset_wild
from models.ColorVidNet import ColorVidNet
from models.ContextualLoss import ContextualLoss, ContextualLoss_forward
from models.FrameColor_stego import frame_colorization
from models.GAN_models import Discriminator_x64
from models.NonlocalNet_stego import (NonlocalWeightedAverage, VGG19_pytorch,
                                WarpNet, WeightedAverage,
                                WeightedAverage_color)
from models.fid_new import get_fid
from models.classification import get_top1_5
from tensorboardX import SummaryWriter
from utils.util import (batch_lab2rgb_transpose_mc,lab2rgb_transpose_mc, feature_normalize, l1_loss_my,
                        mkdir_if_not, mse_loss, parse, tensor_lab2rgb,
                        uncenter_l, weighted_l1_loss, weighted_mse_loss,save_frames)
from utils.util_distortion import (CenterPad_threshold, Normalize, RGB2Lab,CenterPad,
                                   ToTensor)
from utils.util_tensorboard import TBImageRecorder, value_logger
from stego_models.stego_segmentation import STEGO_seg
#from multiprocessing import set_start_method
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from traceback import print_exc

cv2.setNumThreads(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/home/zhanbo/remote/video/video_pair3/", type=str)
parser.add_argument("--data_videoclip", default="/dataset/videvo/test/imgs/CoupleRidingMotorbike", type=str)
parser.add_argument("--data_root_imagenet", default="/dataset/ImageNet_train/", type=str)	
parser.add_argument("--data_root_imagenet10k", default="/dataset/ImageNet_val/val_10000", type=str)	 #need to prepare dataset  
parser.add_argument("--data_root_wild", default="/dataset/from_paper/SSCN", type=str)
parser.add_argument("--gpu_ids", type=str, default="0,1", help="separate by comma")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--image_size", type=int, default=[256, 256])
parser.add_argument("--ic", type=int, default=4)
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--num_class", type=int, default=27)

parser.add_argument("--resume_epoch", type=int, default=0)
parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--resume_iter", type=int, default=120000)
parser.add_argument("--load_pretrained_model", type=bool, default=True)
parser.add_argument("--strict_load", type=bool, default=False)
parser.add_argument("--val_only", type=bool, default=False)
parser.add_argument("--val_wild", type=bool, default=False)
parser.add_argument("--statistic_only", type=bool, default=False)

parser.add_argument("--lr", type=float, default=2e-5)											#这里的lr设定是无效的，并没有使用
parser.add_argument("--lr_discrim", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--lr_step", type=int, default=20000)
parser.add_argument("--lr_gamma", type=float, default=0.5)


parser.add_argument("--checkpoint_dir", type=str, default="/dataset/checkpoints/spcolor/checkpoints/stego_pro_s")
parser.add_argument("--checkpoint_dir_out", type=str, default="/dataset/checkpoints/spcolor/checkpoints/stego_pro_s")
parser.add_argument("--val_output_path", type=str, default="/dataset/checkpoints/spcolor/SSCN")
parser.add_argument("--log_path", type=str, default="/dataset/checkpoints/spcolor/runs/stego_pro_s")
parser.add_argument("--reference_order", type=int, default=1)
parser.add_argument("--tb_log_step", type=int, default=1000)
parser.add_argument("--save_checkpoint_step", type=int, default=10000)
parser.add_argument("--validation_step", type=int, default=5000)

parser.add_argument("--print_step", type=int, default=10)

parser.add_argument("--real_reference_probability", type=float, default=0.8)
parser.add_argument("--nonzero_placeholder_probability", type=float, default=0.0)
parser.add_argument("--with_bad", type=bool, default=True)
parser.add_argument("--with_mid", type=bool, default=True)

parser.add_argument("--domain_invariant", type=bool, default=False)					   #domain invariant
parser.add_argument("--weigth_l1", type=float, default=2.0)
parser.add_argument("--weight_contextual", type=float, default="0.2")
parser.add_argument("--weight_perceptual", type=float, default="0.01") #0.001
parser.add_argument("--weight_smoothness", type=float, default="2.0")  #5
parser.add_argument("--weight_gan", type=float, default="0.4")      #0 .2
parser.add_argument("--weight_discrim", type=float, default="1.0")   #1.0
parser.add_argument("--weight_nonlocal_smoothness", type=float, default="0.0")
parser.add_argument("--weight_nonlocal_consistent", type=float, default="0.0")
parser.add_argument("--luminance_noise", type=float, default="2.0")
parser.add_argument("--permute_data", type=bool, default=True)
parser.add_argument("--contextual_loss_direction", type=str, default="forward", help="forward or backward matching")
parser.add_argument("--use_masked_percept", type=bool, default=True)
parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')


def image_logger_fn(
    I_current_lab,
    I_reference_lab,
    I_current_lab_predict,
    I_current_nonlocal_lab,
    cluster_preds_current,
    cluster_preds_ref,
    S1,
):
    I_current_image = batch_lab2rgb_transpose_mc(I_current_lab[0:32, 0:1, :, :], I_current_lab[0:32, 1:3, :, :],nrow=4)
    I_reference_image = batch_lab2rgb_transpose_mc(I_reference_lab[0:32, 0:1, :, :], I_reference_lab[0:32, 1:3, :, :],nrow=4)
    I_current_image_predict = batch_lab2rgb_transpose_mc(
        I_current_lab_predict[0:32, 0:1, :, :], I_current_lab_predict[0:32, 1:3, :, :],nrow=4
    )
    I_current_nonlocal_image = batch_lab2rgb_transpose_mc(
        I_current_nonlocal_lab[0:32, 0:1, :, :], I_current_nonlocal_lab[0:32, 1:3, :, :],nrow=4
    )
    I_current_class = stego.got_plot(cluster_preds_current)
    I_current_class_warped = stego.got_plot(cluster_preds_current,cluster_value=S1)
    I_ref_class = stego.got_plot(cluster_preds_ref)

    img_info = {}
    img_info["2_I_current"] = I_current_image
    img_info["3_I_reference"] = I_reference_image
    img_info["5_I_curren_predict"] = I_current_image_predict
    img_info["7_I_current_nonlocal"] = I_current_nonlocal_image
    img_info["8_I_current_class"] = I_current_class
    img_info["9_I_current_class_warped"] = I_current_class_warped
    img_info["10_I_ref_class"] = I_ref_class


    return img_info


def training_logger():
    try:
        if total_iter % opt.print_step == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print("processing time:", elapsed)
            S_wo0 = S1[S1>0]
            print(
                "Epoch %d, Step[%d/%d], lr: %f, lr: %f, total_loss: %.2f , m_s: %.2f/%.2f/%.2f , m_s2: %.2f"
                % (
                    epoch,
                    ((iter + 1) % iter_num_per_epoch),
                    iter_num_per_epoch,
                    step_optim_scheduler_g.get_last_lr()[0],
                    step_optim_scheduler_d.get_last_lr()[0],
                    total_loss.item(),
                    torch.mean(S1).item(),
                    torch.mean(S_wo0).item(),
                    torch.max(S1).item(),
                    torch.mean(S2).item(),
                )
            )
            # print("l1_loss",l1_loss.item(),
            #         "feat_loss", feat_loss.item(),
            #         "contextual_loss_total", contextual_loss_total.item(),
            #         "smoothness_loss", smoothness_loss.item(),
            #         "nonlocal_smoothness_loss", nonlocal_smoothness_loss.item(),
            #         "generator_loss", generator_loss.item(),
            #         "discriminator_loss", discriminator_loss.item(),
            #         "total_loss", total_loss.item(),)

            value_logger(
                tb_writer,
                total_iter,
                loss_info={
                    "l1_loss": l1_loss.item(),
                    "feat_loss": feat_loss.item(),
                    "contextual_loss_total": contextual_loss_total.item(),
                    "smoothness_loss": smoothness_loss.item(),
                    "nonlocal_smoothness_loss": nonlocal_smoothness_loss.item(),
                    "generator_loss": generator_loss.item(),
                    "discriminator_loss": discriminator_loss.item(),
                    "total_loss": total_loss.item(),
                    "fid": fid,
                    "top1": top1,
                    "top5": top5,
                },
            )

        if total_iter % opt.tb_log_step == 2:
            # I_current_nonlocal_lab = torch.cat(
            #     (I_current_l, I_current_nonlocal_lab_predict[:, 1:3, :, :]), dim=1
            # )
            data_queue.put(
                (
                    (
                        I_current_lab.cpu(),
                        I_reference_lab.cpu(),
                        I_current_lab_predict.cpu(),
                        I_current_nonlocal_lab_predict.cpu(),
                        cluster_preds_current.cpu(),
                        cluster_preds_ref.cpu(),
                        S1.cpu(),
                    ),
                    total_iter,
                )
            )

        if total_iter % opt.save_checkpoint_step == 0:
            if 1:
                torch.save(
                    nonlocal_net.module.state_dict(),
                    os.path.join(opt.checkpoint_dir_out, "nonlocal_net_iter_%d.pth") % total_iter,
                )
                torch.save(
                    colornet.module.state_dict(),
                    os.path.join(opt.checkpoint_dir_out, "colornet_iter_%d.pth") % total_iter,
                )
                torch.save(
                    discriminator.module.state_dict(),
                    os.path.join(opt.checkpoint_dir_out, "discriminator_iter_%d.pth") % total_iter,
                )
            else:
                torch.save(
                    nonlocal_net.state_dict(),
                    os.path.join(opt.checkpoint_dir_out, "nonlocal_net_iter_%d.pth") % total_iter,
                )
                torch.save(colornet.state_dict(), 
                    os.path.join(opt.checkpoint_dir_out, "colornet_iter_%d.pth") % total_iter
                )
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(opt.checkpoint_dir_out, "discriminator_iter_%d.pth") % total_iter,
                )

        # save the state for resume
        if total_iter % 2000 == 0:
            print("saving the checkpoint")
            if 1:
                state = {
                    "total_iter": total_iter,
                    "epoch": epoch,
                    "colornet_state": colornet.module.state_dict(),
                    "nonlocal_net_state": nonlocal_net.module.state_dict(),
                    "discriminator_state": discriminator.module.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "optimizer_schedule_g": step_optim_scheduler_g.state_dict(),
                    "optimizer_schedule_d": step_optim_scheduler_g.state_dict(),
                }
            else:
                state = {
                    "total_iter": total_iter,
                    "epoch": epoch,
                    "colornet_state": colornet.state_dict(),
                    "nonlocal_net_state": nonlocal_net.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "optimizer_schedule_g": step_optim_scheduler_g.state_dict(),
                    "optimizer_schedule_d": step_optim_scheduler_d.state_dict(),
                }
            torch.save(state, os.path.join(opt.checkpoint_dir_out, "learning_checkpoint.pth"))
    except Exception as e:
        print("Exception during output")
        print(e)						   #logger


						 #设定GPU
def gpu_setup_DDP():
    cudnn.benchmark = True
    device = torch.device("cuda",opt.local_rank)
    return device		

def worker_init_fn(worker_id):
    return np.random.seed(torch.initial_seed()%(2**31)+worker_id)

def load_data_imagenet():
    if opt.local_rank==0:
        print("initializing dataloader")
    # transforms_video = [
    #     CenterCrop(opt.image_size),
    #     RGB2Lab(),
    #     ToTensor(),
    #     Normalize(),
    # ]
    transforms_imagenet = [CenterPad(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    extra_reference_transform = [
        transform_lib.RandomHorizontalFlip(0.5),
        transform_lib.RandomResizedCrop(480, (0.98, 1.0), ratio=(0.8, 1.2)),   # image size | crop ratio | aspect ratio
    ]
    # train_dataset_video = VideosDataset(
    #     data_root=opt.data_root,
    #     epoch=opt.epoch,
    #     image_size=opt.image_size,
    #     image_transform=transforms.Compose(transforms_video),
    #     real_reference_probability=opt.real_reference_probability,
    #     nonzero_placeholder_probability=opt.nonzero_placeholder_probability,
    # )
    train_dataset_imagenet = VideosDataset_ImageNet(
        data_root=opt.data_root_imagenet,
        image_size=opt.image_size,
        epoch=opt.epoch,
        with_bad=opt.with_bad,
        with_mid=opt.with_mid,
        transforms_imagenet=transforms_imagenet,
        distortion_level=4,
        brightnessjitter=5,
        nonzero_placeholder_probability=opt.nonzero_placeholder_probability,
        extra_reference_transform=extra_reference_transform,
        real_reference_probability=opt.real_reference_probability,
    )

    #imagenet_training_length = len(train_dataset_imagenet)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_imagenet)

    data_loader = DataLoader(
        train_dataset_imagenet,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn = worker_init_fn,
        sampler = train_sampler,
    )
    return train_dataset_imagenet.real_len, data_loader


def load_data_video():
    if opt.local_rank==0:
        print("initializing dataloader")

    transforms_imagenet = [CenterPad(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]

    train_dataset_imagenet = VideosDataset_videoclip(
        image_size=opt.image_size,
        transforms_imagenet=transforms_imagenet,
        data_root = opt.data_videoclip
    )

    #imagenet_training_length = len(train_dataset_imagenet)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_imagenet)

    data_loader = DataLoader(
        train_dataset_imagenet,
        batch_size=opt.batch_size + 6,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn = worker_init_fn,
        sampler = train_sampler,
    )
    return  data_loader


def load_data_imagenet10k():
    if opt.local_rank==0:
        print("initializing dataloader")

    transforms_imagenet = [CenterPad(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    if opt.val_wild:
            train_dataset_imagenet = VideosDataset_wild(
                    data_root=opt.data_root_wild,
                    image_size=opt.image_size,
                    transforms_imagenet=transforms_imagenet,
                )
    else:
            train_dataset_imagenet = VideosDataset_ImageNet_10k(
                data_root=opt.data_root_imagenet10k,
                image_size=opt.image_size,
                transforms_imagenet=transforms_imagenet,
                reference_order=opt.reference_order,
            )

    #imagenet_training_length = len(train_dataset_imagenet)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_imagenet)

    data_loader = DataLoader(
        train_dataset_imagenet,
        batch_size=opt.batch_size + 6,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn = worker_init_fn,
        sampler = train_sampler,
    )
    return  data_loader

def define_loss():									 #定义loss
    if opt.local_rank==0:
        print("defining loss")
    # ab_criterion = nn.SmoothL1Loss().to(device)
    # nonlocal_criterion = nn.SmoothL1Loss().to(device)
    # feat_l2_criterion = nn.MSELoss().to(device)
    # feat_l1_criterion = nn.SmoothL1Loss().to(device)
    contextual_loss = ContextualLoss().to(device)						   #contexture loss 自己定义的
    contextual_forward_loss = ContextualLoss_forward().to(device)
    #BCE_stable = nn.BCEWithLogitsLoss().to(device)
    return contextual_loss, contextual_forward_loss


def define_optimizer():
    if opt.local_rank==0:
        print("defining optimizer")
    optimizer_g = optim.Adam(
        [{"params": nonlocal_net.parameters(), "lr": opt.lr}, {"params": colornet.parameters(), "lr": 20 * opt.lr}],
        betas=(0.5, 0.999),
        eps=1e-5,
        amsgrad=True,
    )
    optimizer_d = optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()), lr=opt.lr_discrim, betas=(0.5, 0.999)
    )
    return optimizer_g, optimizer_d						  #定义优化器


def resume_model():									   #继续学习 ，加载参数
    if opt.local_rank==0:
        print("resuming the learning")
    if opt.resume_iter:
        total_iter = opt.resume_iter
        epoch = 0
        checkpoint = torch.load(os.path.join(opt.checkpoint_dir, "nonlocal_net_iter_%d.pth" % total_iter),map_location='cpu')
        nonlocal_net.module.load_state_dict(checkpoint)
        checkpoint = torch.load(os.path.join(opt.checkpoint_dir, "colornet_iter_%d.pth" % total_iter),map_location='cpu')
        colornet.module.load_state_dict(checkpoint)
        checkpoint = torch.load(os.path.join(opt.checkpoint_dir, "discriminator_iter_%d.pth" % total_iter),map_location='cpu')
        discriminator.module.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(os.path.join(opt.checkpoint_dir, "learning_checkpoint.pth"),map_location='cpu')
        total_iter = checkpoint["total_iter"]
        epoch = checkpoint["epoch"]
        colornet.module.load_state_dict(checkpoint["colornet_state"])
        nonlocal_net.module.load_state_dict(checkpoint["nonlocal_net_state"])
        discriminator.module.load_state_dict(checkpoint["discriminator_state"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        step_optim_scheduler_g.load_state_dict(checkpoint["optimizer_schedule_g"])
        step_optim_scheduler_d.load_state_dict(checkpoint["optimizer_schedule_d"])
    return total_iter,epoch


def to_device(
    colornet,
    nonlocal_net,
    discriminator,
    vggnet,
    contextual_loss,
    contextual_forward_loss,
    weighted_layer_color,
    nonlocal_weighted_layer,
    instancenorm,
):
    print("moving models to device")
    colornet = torch.nn.DataParallel(colornet.to(device), device_ids=opt.gpu_ids)
    nonlocal_net = torch.nn.DataParallel(nonlocal_net.to(device), device_ids=opt.gpu_ids)
    discriminator = torch.nn.DataParallel(discriminator.to(device), device_ids=opt.gpu_ids)
    vggnet = torch.nn.DataParallel(vggnet.to(device), device_ids=opt.gpu_ids)
    contextual_loss = torch.nn.DataParallel(contextual_loss.to(device), device_ids=opt.gpu_ids)
    contextual_forward_loss = torch.nn.DataParallel(contextual_forward_loss.to(device), device_ids=opt.gpu_ids)
    weighted_layer_color = torch.nn.DataParallel(weighted_layer_color.to(device), device_ids=opt.gpu_ids)
    nonlocal_weighted_layer = torch.nn.DataParallel(nonlocal_weighted_layer.to(device), device_ids=opt.gpu_ids)
    instancenorm = torch.nn.DataParallel(instancenorm.to(device), device_ids=opt.gpu_ids)

    # vggnet = vggnet.to(device)

    # weighted_layer_color = weighted_layer_color.to(device)
    # nonlocal_weighted_layer =nonlocal_weighted_layer.to(device)
    # instancenorm =instancenorm.to(device)

    return (
        vggnet,
        nonlocal_net,
        colornet,
        discriminator,
        instancenorm,
        contextual_loss,
        contextual_forward_loss,
        weighted_layer_color,
        nonlocal_weighted_layer,
    )									 #to gpu


def to_device_DDP(
    colornet,
    nonlocal_net,
    discriminator,
    vggnet,
    contextual_loss,
    contextual_forward_loss,
    weighted_layer_color,
    nonlocal_weighted_layer,
    instancenorm,
    downsampling_by2,
    stego
):
    if opt.local_rank==0:
        print("moving models to device DDP")
    colornet = torch.nn.parallel.DistributedDataParallel(colornet.to(device), device_ids=[opt.local_rank])
    nonlocal_net = torch.nn.parallel.DistributedDataParallel(nonlocal_net.to(device), device_ids=[opt.local_rank])
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator.to(device), device_ids=[opt.local_rank])
    #vggnet = torch.nn.parallel.DistributedDataParallel(vggnet.to(device), device_ids=[opt.local_rank])
    #contextual_loss = torch.nn.parallel.DistributedDataParallel(contextual_loss.to(device), device_ids=[opt.local_rank])
    #contextual_forward_loss = torch.nn.parallel.DistributedDataParallel(contextual_forward_loss.to(device), device_ids=[opt.local_rank])
    #weighted_layer_color = torch.nn.parallel.DistributedDataParallel(weighted_layer_color.to(device), device_ids=[opt.local_rank])
    #nonlocal_weighted_layer = torch.nn.parallel.DistributedDataParallel(nonlocal_weighted_layer.to(device), device_ids=[opt.local_rank])
    #instancenorm = torch.nn.parallel.DistributedDataParallel(instancenorm.to(device), device_ids=[opt.local_rank])
    #downsampling_by2 = torch.nn.parallel.DistributedDataParallel(downsampling_by2.to(device), device_ids=[opt.local_rank])
    vggnet = vggnet.to(device)
    contextual_loss = contextual_loss.to(device)
    contextual_forward_loss = contextual_forward_loss.to(device)
    weighted_layer_color = weighted_layer_color.to(device)
    nonlocal_weighted_layer =nonlocal_weighted_layer.to(device)
    instancenorm =instancenorm.to(device)
    downsampling_by2 = downsampling_by2.to(device)
    stego.model = stego.model.to(device)
    stego.model.test_cluster_metrics.got_easy_assignments()

    return (
        vggnet,
        nonlocal_net,
        colornet,
        discriminator,
        instancenorm,
        contextual_loss,
        contextual_forward_loss,
        weighted_layer_color,
        nonlocal_weighted_layer,
        downsampling_by2,
        stego
    )									 #to gpu


def loss_init():									 #初始化loss为0
    if opt.local_rank==0:
        print("initializing losses")
    zero_loss = torch.Tensor([0]).to(device)
    (
        feat_loss,
        contextual_loss_total,
        smoothness_loss,
        nonlocal_smoothness_loss,
        generator_loss,
        discriminator_loss,
    ) = (zero_loss, zero_loss, zero_loss, zero_loss, zero_loss, zero_loss)

    return (
        feat_loss,
        contextual_loss_total,
        smoothness_loss,
        nonlocal_smoothness_loss,
        generator_loss,
        discriminator_loss,
    )


def video_colorization():
    # colorization for the current frame
    I_current_ab_predict , out_tensor_warp,_ = frame_colorization(
        I_current_lab,
        I_reference_lab,
        cluster_value_current,
        cluster_value_ref,
        cluster_preds_current,
        cluster_preds_ref,
        features_B,
        vggnet,
        nonlocal_net,
        colornet,
        feature_noise=0,
        luminance_noise=opt.luminance_noise,
    )
    I_current_nonlocal_lab_predict = torch.cat((I_current_l,out_tensor_warp[:,0:2,:,:]),dim=1)
    S1 = out_tensor_warp[:,2:3,:,:]

    return I_current_ab_predict ,I_current_nonlocal_lab_predict.detach(),S1.detach()

def validate_video():
    time_stego=[]
    time_nonlocal=[]
    time_resnet=[]
    time_colornet=[]
    _t_stego=0
    _t_nonlocal=0
    _t_resnet=0
    _t_colornet=0
    val_dataset = load_data_video()
    val_output_path = os.path.join(opt.val_output_path)
    try:
        for iter,data in enumerate(tqdm(val_dataset)):
            (
                I_current_lab,
                I_reference_lab,
                I_current_rgb,
                I_reference_rgb,
                namelist
            ) = data
            #print("got data!")
            I_current_lab = I_current_lab.cuda(non_blocking=True)
            I_reference_lab = I_reference_lab.cuda(non_blocking=True)
            I_current_rgb = I_current_rgb.cuda(non_blocking=True)
            I_reference_rgb = I_reference_rgb.cuda(non_blocking=True)

            I_current_l = I_current_lab[:, 0:1, :, :]										  # l a b -- 0 1 2
            I_current_ab = I_current_lab[:, 1:3, :, :]

            # I_reference_l = I_reference_lab[:, 0:1, :, :]
            # I_reference_ab = I_reference_lab[:, 1:3, :, :]
            #I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))				 #uncenter
            ablation_time=time.time()
            features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            end_time=time.time()
            _t_resnet_1 = end_time-ablation_time
            ###### COLORIZATION ######						###### COLORIZATION ######
            ablation_time=time.time()
            cluster_value_current,cluster_preds_current = stego.my_app(I_current_rgb,mode_unsupervised=True)  # 8 256 256
            cluster_value_ref,cluster_preds_ref = stego.my_app(I_reference_rgb,mode_unsupervised=True)
            end_time=time.time()
            _t_stego = end_time-ablation_time
            time_stego.append(_t_stego)

            with torch.no_grad():
                I_current_ab_predict , _,time_list = frame_colorization(
                I_current_lab,
                I_reference_lab,
                cluster_value_current,
                cluster_value_ref,
                cluster_preds_current,
                cluster_preds_ref,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                luminance_noise=0,
                joint_training = False
                )
            I_current_l = I_current_l.cpu()
            I_current_ab_predict = I_current_ab_predict.cpu()
            _t_resnet = time_list[0] + _t_resnet_1
            _t_nonlocal=time_list[1]
            _t_colornet=time_list[2]
            time_resnet.append(_t_resnet)
            time_nonlocal.append(_t_nonlocal)
            time_colornet.append(_t_colornet)
            #print("corr time:",_t_stego,_t_resnet,_t_resnet_1,_t_nonlocal,_t_colornet)
            #print("average time:",np.mean(time_stego),np.mean(time_resnet),np.mean(time_nonlocal),np.mean(time_colornet))
            _t_stego = 0
            _t_resnet = 0
            _t_nonlocal = 0
            _t_colornet = 0
            if not opt.statistic_only:
                for i in range(len(I_current_l)):
                    I_current_rgb = lab2rgb_transpose_mc(I_current_l[i], I_current_ab_predict[i])
                    save_frames(I_current_rgb, val_output_path, image_name = namelist[i])
                    # I_current_rgb_ori = lab2rgb_transpose_mc(I_current_l[i], I_current_ab[i])
                    # save_frames(I_current_rgb_ori, "/dataset/ImageNet_val/val_10000/croped_imgs", image_name = namelist[i])
    
        torch.distributed.barrier()
        # if dist.get_rank()==0:
        #     fid = get_fid(os.path.join(opt.data_root_imagenet10k,"croped_imgs"),val_output_path,8)
        #     print("fid:",fid)
        #     top1,top5 = get_top1_5(val_output_path)
        # else:
        #     top1,top5 = [0,0]
        #     fid = 0
        # torch.distributed.barrier()
    # except ChildFailedError as e:
    #     print("ChildFatherError: --fid--")
    #     print(e)
    #     fid = 0
    #     top1,top5 = [0,0]
    except Exception as e:
        print("problem in validation: --fid--")
        print(e)
        fid = 0
        top1,top5 = [0,0]

    #torch.distributed.barrier()
    return 0,0,0

def validate_wild():
    
    val_dataset = load_data_imagenet10k()
    val_output_path = os.path.join(opt.val_output_path)
    try:
        for iter,data in enumerate(tqdm(val_dataset)):
            (
                I_current_lab,
                I_reference_lab,
                I_current_rgb,
                I_reference_rgb,
                namelist
            ) = data
            #print("got data!")
            I_current_lab = I_current_lab.cuda(non_blocking=True)
            I_reference_lab = I_reference_lab.cuda(non_blocking=True)
            I_current_rgb = I_current_rgb.cuda(non_blocking=True)
            I_reference_rgb = I_reference_rgb.cuda(non_blocking=True)

            I_current_l = I_current_lab[:, 0:1, :, :]										  # l a b -- 0 1 2
            I_current_ab = I_current_lab[:, 1:3, :, :]

            features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            ###### COLORIZATION ######						###### COLORIZATION ######
            cluster_value_current,cluster_preds_current = stego.my_app(I_current_rgb,mode_unsupervised=True)  # 8 256 256
            cluster_value_ref,cluster_preds_ref = stego.my_app(I_reference_rgb,mode_unsupervised=True,cmp=False)

            with torch.no_grad():
                I_current_ab_predict , I_ab_warped,time_list = frame_colorization(
                I_current_lab,
                I_reference_lab,
                cluster_value_current,
                cluster_value_ref,
                cluster_preds_current,
                cluster_preds_ref,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                luminance_noise=0,
                joint_training = False
                )
            I_current_l = I_current_l.cpu()
            I_current_ab_predict = I_current_ab_predict.cpu()

            for i in range(len(I_current_l)):
                I_current_rgb = lab2rgb_transpose_mc(I_current_l[i], I_current_ab_predict[i])
                save_frames(I_current_rgb, val_output_path, image_name = namelist[i])

    
        torch.distributed.barrier()

    except Exception as e:
        print("problem in validation: --fid--")
        print_exc()

    #torch.distributed.barrier()
    return 


def validate():
    time_stego=[]
    time_nonlocal=[]
    time_resnet=[]
    time_colornet=[]
    _t_stego=0
    _t_nonlocal=0
    _t_resnet=0
    _t_colornet=0
    val_dataset = load_data_imagenet10k()
    val_output_path = os.path.join(opt.val_output_path,"%s" % opt.reference_order)
    mkdir_if_not(val_output_path)
    try:
        for iter,data in enumerate(tqdm(val_dataset)):
            (
                I_current_lab,
                I_reference_lab,
                I_current_rgb,
                I_reference_rgb,
                namelist
            ) = data
            #print("got data!")
            I_current_lab = I_current_lab.cuda(non_blocking=True)
            I_reference_lab = I_reference_lab.cuda(non_blocking=True)
            I_current_rgb = I_current_rgb.cuda(non_blocking=True)
            I_reference_rgb = I_reference_rgb.cuda(non_blocking=True)

            I_current_l = I_current_lab[:, 0:1, :, :]										  # l a b -- 0 1 2
            I_current_ab = I_current_lab[:, 1:3, :, :]

            # I_reference_l = I_reference_lab[:, 0:1, :, :]
            # I_reference_ab = I_reference_lab[:, 1:3, :, :]
            #I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))				 #uncenter
            ablation_time=time.time()
            features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            end_time=time.time()
            _t_resnet_1 = end_time-ablation_time
            ###### COLORIZATION ######						###### COLORIZATION ######
            ablation_time=time.time()
            cluster_value_current,cluster_preds_current = stego.my_app(I_current_rgb,mode_unsupervised=True)  # 8 256 256
            cluster_value_ref,cluster_preds_ref = stego.my_app(I_reference_rgb,mode_unsupervised=True,cmp=False)
            end_time=time.time()
            _t_stego = end_time-ablation_time
            time_stego.append(_t_stego)

            with torch.no_grad():
                I_current_ab_predict , I_ab_warped,time_list = frame_colorization(
                I_current_lab,
                I_reference_lab,
                cluster_value_current,
                cluster_value_ref,
                cluster_preds_current,
                cluster_preds_ref,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                luminance_noise=0,
                joint_training = False
                )
            I_current_l = I_current_l.cpu()
            I_current_ab_predict = I_current_ab_predict.cpu()
            I_ab_warped = I_ab_warped[:,0:2,:,:].cpu()
            _t_resnet = time_list[0] + _t_resnet_1
            _t_nonlocal=time_list[1]
            _t_colornet=time_list[2]
            time_resnet.append(_t_resnet)
            time_nonlocal.append(_t_nonlocal)
            time_colornet.append(_t_colornet)
            #print("corr time:",_t_stego,_t_resnet,_t_resnet_1,_t_nonlocal,_t_colornet)
            #print("average time:",np.mean(time_stego),np.mean(time_resnet),np.mean(time_nonlocal),np.mean(time_colornet))
            _t_stego = 0
            _t_resnet = 0
            _t_nonlocal = 0
            _t_colornet = 0
            if not opt.statistic_only:
                for i in range(len(I_current_l)):
                    I_current_rgb = lab2rgb_transpose_mc(I_current_l[i], I_current_ab_predict[i])
                    save_frames(I_current_rgb, val_output_path, image_name = namelist[i])
                    #I_current_rgb_warped = lab2rgb_transpose_mc(I_current_l[i], I_ab_warped[i])
                    #save_frames(I_current_rgb_warped, "/dataset/temp", image_name = "warped_color.png")
                    # I_current_rgb_ori = lab2rgb_transpose_mc(I_current_l[i], I_current_ab[i])
                    # save_frames(I_current_rgb_ori, "/dataset/ImageNet_val/val_10000/croped_imgs", image_name = namelist[i])
    
        torch.distributed.barrier()
        if dist.get_rank()==0:
            fid = get_fid(os.path.join(opt.data_root_imagenet10k,"croped_imgs"),val_output_path,8)
            print("fid:",fid)
            top1,top5 = get_top1_5(val_output_path)
        else:
            top1,top5 = [0,0]
            fid = 0
        torch.distributed.barrier()
    # except ChildFailedError as e:
    #     print("ChildFatherError: --fid--")
    #     print(e)
    #     fid = 0
    #     top1,top5 = [0,0]
    except Exception as e:
        print("problem in validation: --fid--")
        print_exc()
        fid = 0
        top1,top5 = [0,0]

    #torch.distributed.barrier()
    return fid,top1,top5

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method("fork", force=True)
    #set_start_method('fork')
    #os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    opt = parse(parser)
    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend='nccl')
    #opt.data_root = opt.data_root.split(",")[0]
    #opt.data_root_imagenet = opt.data_root_imagenet.split(",")[0]											   #make dir
    if opt.local_rank == 0:
        mkdir_if_not(opt.checkpoint_dir_out)
        mkdir_if_not(opt.val_output_path)
    #mkdir_if_not("./runs/")

    device = gpu_setup_DDP()
    dataset_real_len, data_loader = load_data_imagenet()			   #load data
    iter_num_per_epoch = len(data_loader) #// opt.batch_size
    if dist.get_rank() == 0:
        tb_writer = SummaryWriter(log_path=opt.log_path)
        data_queue = queue.Queue()
        tb_image_reorder = TBImageRecorder(tb_writer, image_logger_fn, data_queue)
        tb_image_reorder.start()

    # define network																						       #network
    nonlocal_net = WarpNet(opt.batch_size)
    colornet = ColorVidNet(opt.ic)
    discriminator = Discriminator_x64(in_size=3,ndf=64)
    if opt.num_class < 27:
        print("using cra in segmantation, num class:",opt.num_class)
        cra = True
    else:
        cra = False
    stego = STEGO_seg(cra,opt.num_class)
    for param in stego.par_model.parameters():
        param.requires_grad = False

    weighted_layer = WeightedAverage()
    weighted_layer_color = WeightedAverage_color()															  #weighted_layer
    nonlocal_weighted_layer = NonlocalWeightedAverage()
    instancenorm = nn.InstanceNorm2d(512, affine=False)

    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("/dataset/checkpoints/spcolor/checkpoints/video_moredata_l1/vgg19_conv.pth",map_location='cpu'))
    vggnet.eval()																							  #加载VGG19并固定参数
    for param in vggnet.parameters():
        param.requires_grad = False

    # load pre-trained model
    if opt.load_pretrained_model and not opt.resume:
        nonlocal_pretain_path = os.path.join("/dataset/checkpoints/spcolor/checkpoints/video_moredata_l1/", "nonlocal_net_iter_76000.pth")			  #这个加载的是什么？
        nonlocal_net.load_state_dict(torch.load(nonlocal_pretain_path,map_location='cpu'),strict=opt.strict_load)
        color_test_path = "/dataset/checkpoints/spcolor/checkpoints/video_moredata_l1/" + "colornet_iter_76000.pth"	
        color_test = torch.load(color_test_path,map_location='cpu')
        #del color_test["conv1_1.0.weight"]
        colornet.load_state_dict(color_test,strict=opt.strict_load)
        discriminator_pretain_path = os.path.join("/dataset/checkpoints/spcolor/checkpoints/video_moredata_l1/", "discriminator_iter_76000.pth")
        discriminator_pretain = torch.load(discriminator_pretain_path,map_location='cpu')
        #del discriminator_pretain['layer1']
        discriminator.load_state_dict(discriminator_pretain,strict=opt.strict_load)

    # define loss function
    contextual_loss, contextual_forward_loss = define_loss()															  #定义loss
    downsampling_by2 = nn.AvgPool2d(kernel_size=2)

    # move to GPU processing
    (
        vggnet,
        nonlocal_net,
        colornet,
        discriminator,
        instancenorm,
        contextual_loss,
        contextual_forward_loss,
        weighted_layer_color,
        nonlocal_weighted_layer,
        downsampling_by2,
        stego
    ) = to_device_DDP(
        colornet,
        nonlocal_net,
        discriminator,
        vggnet,
        contextual_loss,
        contextual_forward_loss,
        weighted_layer_color,
        nonlocal_weighted_layer,
        instancenorm,
        downsampling_by2,
        stego
    )

    (
        feat_loss,
        contextual_loss_total,
        smoothness_loss,
        nonlocal_smoothness_loss,
        generator_loss,
        discriminator_loss,
    ) = loss_init()


    # define optimizer
    optimizer_g, optimizer_d = define_optimizer()
    step_optim_scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=opt.lr_step, gamma=opt.lr_gamma)		  #定义优化器	   generator discriminator
    step_optim_scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=opt.lr_step, gamma=opt.lr_gamma)

    if opt.resume:
        total_iter,opt.resume_epoch=resume_model()
    else:
        total_iter = 0

    # dataset info
    #iter_num_per_epoch = dataset_training_length // opt.batch_size
    #total_iter = opt.resume_epoch * iter_num_per_epoch
    # print(
    #     "train_dataset info,  real_len: %d, epoch_len: %d, iter_num_per_epoch: %d"
    #     % (dataset_training_length, len(train_dataset_video) + len(train_dataset_imagenet), iter_num_per_epoch)
    # )
    fid = 0
    top1,top5=[0,0]

    if opt.local_rank==0:
        n_parameters_non = sum(p.numel() for n,p in nonlocal_net.named_parameters())
        n_parameters_vgg = sum(p.numel() for n,p in vggnet.named_parameters())
        n_parameters_colornet = sum(p.numel() for n,p in colornet.named_parameters())
        n_parameters_stego = sum(p.numel() for n,p in stego.par_model.named_parameters())
        print("number of paras:",n_parameters_non+n_parameters_vgg+n_parameters_colornet,n_parameters_stego)

    if opt.val_only:
        print("start validation!")
        try:
            if opt.val_wild:
                validate_wild()
            else:
                fid,top1,top5 = validate()
        except Exception as e:
            print("---fail validation---")
            print_exc()
        print("validation done!")
        os._exit(1)

    # %% Training
    if opt.local_rank==0:
        print("start training")
    for epoch in range(opt.resume_epoch,opt.epoch):
        if opt.local_rank==0:
            print("epoch %d" % epoch)
        start_time = time.time()
        data_loader.sampler.set_epoch(epoch)
        epoch_start = time.time()
        for iter, data in enumerate(data_loader):					 #每个iter读取的数据
            #print("in training!",len(data_loader))
            total_iter += 1

            ###### LOADING DATA SAMPLE ######
            (
                I_current_lab,
                I_reference_lab,
                I_current_rgb,
                I_reference_rgb,
                self_ref_flag,
            ) = data
            #print("got data!")
            I_current_lab = I_current_lab.cuda(non_blocking=True)
            I_reference_lab = I_reference_lab.cuda(non_blocking=True)
            I_current_rgb = I_current_rgb.cuda(non_blocking=True)
            I_reference_rgb = I_reference_rgb.cuda(non_blocking=True)
            self_ref_flag = self_ref_flag.cuda(non_blocking=True)

            I_current_l = I_current_lab[:, 0:1, :, :]										  # l a b -- 0 1 2
            I_current_ab = I_current_lab[:, 1:3, :, :]

            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            #I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))				 #uncenter
            features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            ###### stego clusters#####
            cluster_value_current,cluster_preds_current = stego.my_app(I_current_rgb)  # 8 256 256
            cluster_value_ref,cluster_preds_ref = stego.my_app(I_reference_rgb)

            ###### COLORIZATION ######						###### COLORIZATION ######
            (
                I_current_ab_predict,
                I_current_nonlocal_lab_predict,
                S1
            ) = video_colorization()
            #print("colorized!")
            ###### UPDATE DISCRIMINATOR ######				 ###### UPDATE DISCRIMINATOR ######
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            if opt.weight_gan > 0:

                fake_data_lab = torch.cat(
                    (uncenter_l(I_current_l), I_current_ab_predict), dim=1
                )
                real_data_lab = torch.cat((uncenter_l(I_current_l), I_current_ab), dim=1)

                if opt.permute_data:
                    batch_index = torch.arange(-1, opt.batch_size - 1, dtype=torch.long)
                    real_data_lab = real_data_lab[batch_index, ...]

                y_pred_fake, feature_pred_fake = discriminator(fake_data_lab.detach())
                y_pred_real, feature_pred_real = discriminator(real_data_lab.detach())

                y = torch.ones_like(y_pred_real)
                y2 = torch.zeros_like(y_pred_real)
                discriminator_loss = (
                    torch.mean((y_pred_real - torch.mean(y_pred_fake) - y) ** 2)
                    + torch.mean((y_pred_fake - torch.mean(y_pred_real) + y) ** 2)
                ) / 2 * opt.weight_discrim
                discriminator_loss.backward()
                optimizer_d.step()

            ###### UPDATE GENERATOR ######
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # extract vgg features for both output and original image
            I_predict_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), I_current_ab_predict), dim=1))
            predict_relu1_1, predict_relu2_1, predict_relu3_1, predict_relu4_1, predict_relu5_1 = vggnet(
                I_predict_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
            )

            I_current_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), I_current_ab), dim=1))
            A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vggnet(
                I_current_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
            )
            B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B

            ###### LOSS COMPUTE ######
            # l1 loss
            if opt.weigth_l1 > 0:
                sample_weights = (self_ref_flag[:, 1:3, :, :]) / (sum(self_ref_flag[:, 0, 0, 0]) + 1e-5)
                l1_loss = weighted_l1_loss(I_current_ab_predict, I_current_ab, sample_weights) * opt.weigth_l1

            # generator loss
            if opt.weight_gan > 0:
                y_pred_fake, feature_pred_fake = discriminator(fake_data_lab)
                y_pred_real, feature_pred_real = discriminator(real_data_lab)
                generator_loss = (
                    (
                        torch.mean((y_pred_real - torch.mean(y_pred_fake) + y) ** 2)
                        + torch.mean((y_pred_fake - torch.mean(y_pred_real) - y) ** 2)
                    )
                    / 2
                    * opt.weight_gan
                )

            # feature loss
            if opt.domain_invariant:
                feat_loss = (
                    mse_loss(instancenorm(predict_relu5_1), instancenorm(A_relu5_1.detach()))
                    * opt.weight_perceptual
                    * 1e5
                    * 0.2
                )
            else:
                if opt.use_masked_percept:
                    # S2 = S1 / 0.5
                    # S2[S2<0.4]=0
                    # S2[S2>0.9]=0.9
                    S2 = 1 - S1
                    S2 = F.interpolate(S2,size=A_relu5_1.shape[-2:])
                    feat_loss = weighted_mse_loss(predict_relu5_1,A_relu5_1.detach(),S2) * opt.weight_perceptual
                else:
                    S2=torch.tensor([0])
                    feat_loss = mse_loss(predict_relu5_1, A_relu5_1.detach()) * opt.weight_perceptual

            # contextual loss
            if opt.contextual_loss_direction == "backward":
                contextual_style5_1 = torch.mean(contextual_loss(predict_relu5_1, B_relu5_1.detach())) * 8
                contextual_style4_1 = torch.mean(contextual_loss(predict_relu4_1, B_relu4_1.detach())) * 4
                contextual_style3_1 = (
                    torch.mean(contextual_loss(downsampling_by2(predict_relu3_1), downsampling_by2(B_relu3_1.detach()))) * 2
                )
            else:
                contextual_style5_1 = torch.mean(contextual_forward_loss(predict_relu5_1, B_relu5_1.detach())) * 8
                contextual_style4_1 = torch.mean(contextual_forward_loss(predict_relu4_1, B_relu4_1.detach())) * 4
                contextual_style3_1 = (
                    torch.mean(
                        contextual_forward_loss(downsampling_by2(predict_relu3_1), downsampling_by2(B_relu3_1.detach()))
                    )
                    * 2
                )
            if opt.weight_contextual > 0:
                contextual_loss_total = (
                    contextual_style5_1 + contextual_style4_1 + contextual_style3_1
                ) * opt.weight_contextual

            # smoothness loss
            if opt.weight_smoothness > 0:
                scale_factor = 1
                I_current_lab_predict = torch.cat((I_current_l, I_current_ab_predict), dim=1)
                IA_ab_weighed = weighted_layer_color(
                    I_current_lab, I_current_lab_predict, patch_size=3, alpha=2, scale_factor=scale_factor
                )
                smoothness_loss = (
                    mse_loss(nn.functional.interpolate(I_current_ab_predict, scale_factor=scale_factor), IA_ab_weighed)
                    * opt.weight_smoothness
                )

            if opt.weight_nonlocal_smoothness > 0:
                scale_factor = 0.25
                alpha_nonlocal_smoothness = 0.5
                nonlocal_smooth_feature = feature_normalize(A_relu2_1)
                I_current_lab_predict = torch.cat((I_current_l, I_current_ab_predict), dim=1)
                I_current_ab_weighted_nonlocal = nonlocal_weighted_layer(
                    I_current_lab_predict,
                    nonlocal_smooth_feature.detach(),
                    patch_size=3,
                    alpha=alpha_nonlocal_smoothness,
                    scale_factor=scale_factor,
                )
                nonlocal_smoothness_loss = (
                    mse_loss(
                        nn.functional.interpolate(I_current_ab_predict, scale_factor=scale_factor),
                        I_current_ab_weighted_nonlocal,
                    )
                    * opt.weight_nonlocal_smoothness
                )


            # total loss
            total_loss = (
                l1_loss
                + feat_loss
                + contextual_loss_total
                + smoothness_loss
                + nonlocal_smoothness_loss
                + generator_loss
            )
            total_loss.backward()
            optimizer_g.step()
            
            if total_iter % opt.validation_step == 0:
                print("start validation!")
                torch.cuda.empty_cache()
                try:
                    fid,top1,top5 = validate()
                except Exception as e:
                    print("---fail validation---")
                    print(e)
                #print("result fid:",fid)
            if dist.get_rank() == 0:
                training_logger()
            if total_iter % opt.print_step == 0:
                start_time = time.time()
            step_optim_scheduler_g.step()
            step_optim_scheduler_d.step()
        epoch_time = time.time() - epoch_start
        print("epoch_time:",epoch_time)
        if dist.get_rank() == 0:
            data_queue.put((None, None))
