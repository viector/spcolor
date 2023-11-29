from .modules import *
from .data import *
from collections import defaultdict
#from multiprocessing import get_context  #Pool
import hydra
import seaborn as sns
#import torch.multiprocessing
from crf import dense_crf 
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels

#torch.multiprocessing.set_sharing_strategy('file_system')


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



@hydra.main(config_path="configs", config_name="eval_config.yml")
def stego_seg(cfg: DictConfig) -> None:
    pytorch_data_dir = '/dataset'

    model_path = "../saved_models/cocostuff27_vit_base_5.ckpt"
    model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)

    loader_crop = "center"
    test_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=model.cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(cfg.res, False, loader_crop),
        target_transform=get_transform(cfg.res, True, loader_crop),
        cfg=model.cfg,
    )

    test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                                shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    par_model = model.net

    if model.cfg.dataset_name == "cocostuff27":
        # all_good_images = range(10)
        # all_good_images = range(250)
        # all_good_images = [61, 60, 49, 44, 13, 70] #Failure cases
        all_good_images = [19, 54, 67, 66, 65, 75, 77, 76, 124]  # Main figure
    elif model.cfg.dataset_name == "cityscapes":
        # all_good_images = range(80)
        # all_good_images = [ 5, 20, 56]
        all_good_images = [11, 32, 43, 52]
    else:
        raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))
    #for locating good image in dataloader
    batch_nums = torch.tensor([n // (cfg.batch_size * 2) for n in all_good_images])
    batch_offsets = torch.tensor([n % (cfg.batch_size * 2) for n in all_good_images])

    saved_data = defaultdict(list)
    #with get_context('spawn').Pool(cfg.num_workers + 5) as pool:
    #with Pool(cfg.num_workers + 5) as pool:
    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            feats, code1 = par_model(img)
            # feats, code2 = par_model(img.flip(dims=[3]))
            # code = (code1 + code2.flip(dims=[3])) / 2
            code = code1

            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            #linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
            cluster_probs = model.cluster_probe(code, 2, log_probs=True)

            #linear_preds = linear_probs.argmax(1)
            cluster_preds = cluster_probs.argmax(1)

            #model.test_linear_metrics.update(linear_preds, label)
            #model.test_cluster_metrics.update(cluster_preds, label)
            model.test_cluster_metrics.got_easy_assignments()

            if i in batch_nums:
                matching_offsets = batch_offsets[torch.where(batch_nums == i)]
                for offset in matching_offsets:
                    #saved_data["linear_preds"].append(linear_preds.cpu()[offset].unsqueeze(0))
                    saved_data["cluster_preds"].append(cluster_preds.cpu()[offset].unsqueeze(0))
                    saved_data["label"].append(label.cpu()[offset].unsqueeze(0))
                    saved_data["img"].append(img.cpu()[offset].unsqueeze(0))
    saved_data = {k: torch.cat(v, dim=0) for k, v in saved_data.items()}

    # tb_metrics = {
    #     #**model.test_linear_metrics.compute(),
    #     **model.test_cluster_metrics.compute(),
    # }
    for good_images in batch_list(range(len(all_good_images)), 10):
        for i, img_num in enumerate(good_images):
            plot_img = (prep_for_plot(saved_data["img"][img_num]) * 255).numpy().astype(np.uint8)
            plot_label = (model.label_cmap[saved_data["label"][img_num]]).astype(np.uint8)
            #Image.fromarray(plot_img).save(join(join(result_dir, "img", str(img_num) + ".jpg")))
            #Image.fromarray(plot_label).save(join(join(result_dir, "label", str(img_num) + ".png")))

            plot_cluster = (model.label_cmap[
                model.test_cluster_metrics.map_clusters(
                    saved_data["cluster_preds"][img_num])]) \
                .astype(np.uint8)
            plot_cluster = plot_cluster.squeeze(2)
            Image.fromarray(plot_cluster).save(join(join(result_dir, "cluster_noassign", str(img_num) + ".jpg")))


if __name__ == "__main__":
    prep_args()
    result_dir = "../results/coco"
    stego_seg()
