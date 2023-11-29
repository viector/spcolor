from modules import *
import hydra
import torch.multiprocessing
from PIL import Image
#from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
import torchvision.utils as vutils
import os
#import numpy as np


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = join(root)
        self.transform = transform
        self.images = os.listdir(self.root)
        self.images = sorted(self.images,key=lambda f: int("".join(filter(str.isdigit, f))))

    def __getitem__(self, index):
        image = Image.open(join(self.root, self.images[index])).convert('RGB')  #.convert('L')
        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        # torch.manual_seed(seed)
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)


class STEGO_seg():
    def __init__(self,cra=False,n_clusters=27):
        '''
        cluster modes: Kmeans and Hierarchical
        if post cluster, use cluster label to map the semantic class after classificatio to 27 classes.
        if pre cluster, use new cluster center for classification.
        '''
        model = LitUnsupervisedSegmenter.load_from_checkpoint("/sqchen/code/spcolor/saved_models/cocostuff27_vit_base_5.ckpt")
        #model.eval()
        self.model = model
        self.par_model = model.net
        self.mode = ["Kmeans","pre"] # Kmeans  Hierarchical
        self.map_dict = {}
        if cra:
            cluster_mapping = self.model.redefine_cluster(n_clusters=n_clusters,mode = self.mode)
            self.prepare_post_cluster(cluster_mapping)
        self.model.eval()

    def prepare_post_cluster(self,cluster_mapping):
        if  cluster_mapping is None:  # [16, 16, 26, 23, 22, 17, 25, 24, 19, 11, 21, 18, 8, 12, 13, 10, 15, 14, 5, 6, 2, 7, 9, 4, 3, 1, 0]
            return
        class_dict = {}
        for i in range(27):
            if cluster_mapping[i] not in class_dict:   # 16 in class_dict ?
                class_dict[cluster_mapping[i]]=i       # class_dict: {16: 0}
            else:
                self.map_dict[i]=class_dict[cluster_mapping[i]]  # map_dict: {1:0}
        print("cluster_mapping",cluster_mapping)
        print("class_dict",class_dict)
        print("map_dict",self.map_dict)
        return
    
    def post_cluster(self,cluster_preds):
        for key in self.map_dict:
            cluster_preds[cluster_preds==key] = self.map_dict[key]
        return cluster_preds

    def my_app(self,img,mode_unsupervised = True,cmp=False):
        with torch.no_grad():
            if img.shape[-1]!=256 or img.shape[-2]!=256:
                _img = F.interpolate(img, [256,256], mode='bilinear', align_corners=False)
                feats, code = self.par_model(_img)
                code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
            else:
                feats, code = self.par_model(img)
            
            #linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
            if mode_unsupervised:
                cluster_probs = self.model.cluster_probe(code, 10, soft_probs=True)  # 10 is a temperature coefficient
            else:
                cluster_probs = torch.log_softmax(self.model.linear_probe(code), dim=1)
            cluster_value , cluster_preds = cluster_probs.max(1)
            #print("cluster_value:",cluster_value.shape,cluster_value.min(),cluster_value.mean(),cluster_value.max())
            # cluster_value = torch.ones_like(cluster_value) * 0.9
            # cluster_preds = torch.zeros_like(cluster_preds)
            #cluster_preds = cluster_probs.argmax(1)
            #---------- post processing for class label-----------#
            if self.mode[1] == "post" and cmp:
                cluster_preds = self.post_cluster(cluster_preds)
            #------for class 1------#
            # cluster_preds = torch.zeros_like(cluster_preds)
            # cluster_value = torch.ones_like(cluster_value)
            return cluster_value ,cluster_preds
            # for j in range(img.shape[0]):
            #     #linear_crf = linear_probs[j].argmax(0)
            #     cluster_crf = cluster_probs[j].argmax(0)

            #     new_name = ".".join(name[j].split(".")[:-1]) + ".png"
            #     #Image.fromarray(linear_crf.astype(np.uint8)).save(join(result_dir, "linear", new_name))
            #     Image.fromarray(cluster_crf.astype(np.uint8)).save(join(result_dir, "cluster", new_name))

    def got_plot(self,cluster_preds,nrow = 4,cluster_value=None):
        #print("cluster_preds",cluster_preds.shape)  #torch.Size([8, 256, 256])
        #--------------no lable so equal mapping------------
        #cluster_preds = stego.model.test_cluster_metrics.map_clusters(cluster_preds[0])
        # for i in range(len(cluster_preds)):
        #     clu_pre = cluster_preds[i]
        #print("cluster_preds",cluster_preds.shape)  #torch.Size([256, 256, 1])
        plot_cluster = (self.model.label_cmap[cluster_preds]).astype(np.uint8)
        #print("plot_cluster",plot_cluster.shape)
        #plot_cluster = plot_cluster.squeeze(2)  #plot_cluster (256, 256, 3)
        pred_class = torch.tensor(plot_cluster).permute(0,3,1,2)
        #print("pred_class",pred_class.shape)  #   8 256 256 3 - 8 3 256 256
        if cluster_value is not None:
            if cluster_value.ndim == 3:
                cluster_value = cluster_value.unsqueeze(1)
            pred_class = pred_class * cluster_value
        #print("plot_cluster",plot_cluster.shape)
        #Image.fromarray(plot_cluster).save(join(join(result_dir, str(i) + "_plot.png")))
        grid_class = vutils.make_grid(pred_class, nrow=4).numpy().astype("float64")  # c h w
        return (np.clip(grid_class.transpose((1, 2, 0)), 0, 255)).astype("uint8")
        

def plot_class_colors(stego):
    cluster_preds = np.arange(27)
    cluster_preds = np.tile(cluster_preds,(256,10))
    cluster_preds.sort()
    cluster_preds = torch.tensor(cluster_preds).unsqueeze(0)
    plot_cluster = (stego.model.label_cmap[cluster_preds]).astype(np.uint8)
    plot_cluster = plot_cluster.squeeze(0)  #plot_cluster (256, 256, 3)
    #print("plot_cluster",plot_cluster.shape)
    Image.fromarray(plot_cluster).save(join(join("../results/temp", "semantic_colors.png")))
    print("done plot_class_colors")
    return


if __name__ == "__main__":
    '''
    /dataset/checkpoints/spcolor/val_10000/cra/h27        color results in 27 classes
    /dataset/checkpoints/spcolor/val_10000/cra/ref1/      refrences
    ../results/good_imgs_all_27
    convert L for inputs   X
    cra F
    map F
    '''
    image_dir = "../results/test_imgs" 
    result_dir = "../results/test_imgs_out"
    res = 256
    batch_size = 1
    num_workers = 4
    easy_assignment = True

    dataset = UnlabeledImageFolder(
        root=image_dir,
        transform=get_transform(res, False, "center"),
    )

    loader = DataLoader(dataset, batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    stego = STEGO_seg(cra=False,n_clusters=27)
    stego.model=stego.model.cuda()
    if easy_assignment:
        stego.model.test_cluster_metrics.got_easy_assignments()
    # for i, img in enumerate(tqdm(loader)):
    #     cluster_value, cluster_preds = stego.my_app(img.cuda())
    #     np_grid_class = stego.got_plot(cluster_preds.cpu(),cluster_value=cluster_value.cpu())
    #     Image.fromarray(np_grid_class).save(join(join(result_dir, str(i) + "_grid_plot_v.png")))
    #     break
    os.makedirs(result_dir,exist_ok=True)

    #plot_class_colors(stego)

    for i, img in enumerate(tqdm(loader)):
        cluster_value,cluster_preds = stego.my_app(img.cuda())
        #model.test_cluster_metrics.update(cluster_preds, label)
        #save_cluster_preds = cluster_preds[0].numpy().astype(np.uint8)
        #print("cluster_preds",cluster_preds.shape)  #torch.Size([8, 256, 256])
        #cluster_preds = stego.model.test_cluster_metrics.map_clusters(cluster_preds[0])
        #print("cluster_preds",cluster_preds.shape)  #torch.Size([256, 256, 1])
        print("clusters %d :" % i, set(cluster_preds.view(-1).cpu().numpy()))
        plot_cluster = (stego.model.label_cmap[cluster_preds.cpu()]).astype(np.uint8)
        plot_cluster = plot_cluster.squeeze(0)  #plot_cluster (256, 256, 3)
        #print("plot_cluster",plot_cluster.shape)
        Image.fromarray(plot_cluster).save(join(join(result_dir, str(i) + "_plot.png")))
        #Image.fromarray(save_cluster_preds).save(join(result_dir, str(i) + "_demo.png"))

