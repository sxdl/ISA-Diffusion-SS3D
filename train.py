""" Training (stage 2) function for Diffusion-SS3D

Written by: ChengJu Ho, 2023
Based on: VoteNet, SESS, 3DIoUMatch, DiffusionDet
"""

import os
import sys
import wandb
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from models.votenet_iou_branch import VoteNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "models"))

from pointnet2.pytorch_utils import BNMomentumScheduler

from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from models.loss_helper_labeled import get_labeled_loss
from models.loss_helper_unlabeled import get_unlabeled_loss
from models.loss_helper import get_loss

from models.augment_helper import AugmentHelper

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="votenet", help="Model file name.")
parser.add_argument("--dataset", default="scannet", help="Dataset name")
parser.add_argument("--labeled_sample_list", default="scannetv2_train_0.05.txt", help="Labeled sample list from a certain percentage of training [static]")
parser.add_argument("--detector_checkpoint", default="none")
parser.add_argument("--log_dir", default="./temp", help="Dump dir to save model checkpoint")
parser.add_argument("--num_point", type=int, default=40000, help="Point Number")
parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in Votenet input.")
parser.add_argument("--use_color", action="store_true", help="Use RGB color in Votenet input.")
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument("--num_target", type=int, default=128, help="Proposal number")
parser.add_argument("--vote_factor", type=int, default=1, help="Vote factor")
parser.add_argument("--ap_iou_thresh", type=float, default=0.25, help="AP IoU threshold")
parser.add_argument("--max_epoch", type=int, default=1001, help="Epoch to run")
parser.add_argument("--batch_size", default="4,8", help="Batch Size during training, labeled + unlabeled")
parser.add_argument("--learning_rate", type=float, default=0.005, help="Initial learning rate")
parser.add_argument("--weight_decay", type=float, default=0, help="Optimization L2 weight decay")
parser.add_argument("--bn_decay_step", type=int, default=20, help="Period of BN decay (in epochs)")
parser.add_argument("--bn_decay_rate", type=float, default=0.5, help="Decay rate for BN decay")
parser.add_argument("--lr_decay_steps", default="400, 600, 800, 900", help="When to decay the learning rate (in epochs)")
parser.add_argument("--lr_decay_rates", default="0.3, 0.3, 0.1, 0.1", help="Decay rates for lr decay")
parser.add_argument("--ema_decay", type=float, default=0.999, metavar="ALPHA", help="ema variable decay rate")
parser.add_argument("--unlabeled_loss_weight", type=float, default=2.0, metavar="WEIGHT", help="use unlabeled loss with given weight")
parser.add_argument("--use_iou_for_nms", action="store_true", help="whether use iou to guide test-time nms")
parser.add_argument("--print_interval", type=int, default=25, help="batch interval to print loss")
parser.add_argument("--eval_interval", type=int, default=20, help="epoch interval to evaluate model")
parser.add_argument("--resume", action="store_true", help="resume training instead of just loading a pre-train model")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--conf_thresh", type=float, default=0.05)
parser.add_argument("--view_stats", action="store_true")
parser.add_argument("--opt_rate", type=float, default=1e-3, help="Optimization rate for eval with opt")
parser.add_argument("--opt_step", type=int, default=10, help="Optimization step for eval with opt")
parser.add_argument("--data_ratio", type=float, default=0.05, help="data_ratio")
parser.add_argument("--iterative_train", type=int,default=2, help="Number of forward training passes.")
parser.add_argument("--size_snr_scale", type=float, default=2.0, help="snr_scale for diffusion size.")
parser.add_argument("--label_snr_scale",type=float,default=4.0,help="snr_scale for diffusion label")
parser.add_argument("--timesteps", type=int, default=1000, help="timesteps")
parser.add_argument("--sampling_timesteps", type=int, default=2, help="Num of sampling timesteps for diffusion process.")
parser.add_argument("--use_wandb", action="store_true")

parser.add_argument('--mixup_ratio', type=float, default=0.5)

FLAGS = parser.parse_args()

print(FLAGS)
################################ END ################################

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
print("\n************************** GLOBAL CONFIG BEG **************************")

batch_size_list = [int(x) for x in FLAGS.batch_size.split(",")]
BATCH_SIZE = batch_size_list[0] + batch_size_list[1]
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(",")]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(",")]
assert len(LR_DECAY_STEPS) == len(LR_DECAY_RATES)

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "a")
LOG_FOUT.write(str(FLAGS) + "\n")
if not FLAGS.eval:
    PERFORMANCE_FOUT = open(os.path.join(LOG_DIR, "best.txt"), "w")


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Init datasets and dataloaders
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd.sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset
    from sunrgbd.sunrgbd_ssl_dataset import SunrgbdSSLLabeledDataset, SunrgbdSSLUnlabeledDataset
    from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    LABELED_DATASET = SunrgbdSSLLabeledDataset(labeled_sample_list=FLAGS.labeled_sample_list,
                                               num_points=NUM_POINT,
                                               augment=True,
                                               use_color=FLAGS.use_color,
                                               use_height=(not FLAGS.no_height),
                                               use_v1 = (not FLAGS.use_sunrgbd_v2))
    UNLABELED_DATASET = SunrgbdSSLUnlabeledDataset(labeled_sample_list=FLAGS.labeled_sample_list,
                                                   num_points=NUM_POINT,
                                                   use_color=FLAGS.use_color,
                                                   use_height=(not FLAGS.no_height),
                                                   use_v1=(not FLAGS.use_sunrgbd_v2),
                                                   load_labels=FLAGS.view_stats,
                                                   augment=True)
    TEST_DATASET = SunrgbdDetectionVotesDataset('val',
                                                num_points=NUM_POINT, augment=False,
                                                use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                                use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, "scannet"))
    from scannet.scannet_detection_dataset import ScannetDetectionDataset
    from scannet.scannet_ssl_dataset import ScannetSSLLabeledDataset, ScannetSSLUnlabeledDataset
    from scannet.model_util_scannet import ScannetDatasetConfig

    DATASET_CONFIG = ScannetDatasetConfig()
    LABELED_DATASET = ScannetSSLLabeledDataset(
        labeled_sample_list=FLAGS.labeled_sample_list,
        num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color,
        use_height=(not FLAGS.no_height),
    )
    UNLABELED_DATASET = ScannetSSLUnlabeledDataset(
        labeled_sample_list=FLAGS.labeled_sample_list,
        num_points=NUM_POINT,
        use_color=FLAGS.use_color,
        use_height=(not FLAGS.no_height),
        load_labels=FLAGS.view_stats,
        augment=True,
    )
    TEST_DATASET = ScannetDetectionDataset(
        "val",
        num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color,
        use_height=(not FLAGS.no_height),
    )
else:
    print("Unknown dataset %s. Exiting..." % (FLAGS.dataset))
    exit(-1)
    
log_string("Dataset sizes: labeled-{0}; unlabeled-{1}; VALID-{2}".format(len(LABELED_DATASET), len(UNLABELED_DATASET), len(TEST_DATASET)))

LABELED_DATALOADER = DataLoader(
    LABELED_DATASET,
    batch_size=batch_size_list[0],
    shuffle=True,
    num_workers=batch_size_list[0],
    worker_init_fn=my_worker_init_fn,
)
UNLABELED_DATALOADER = DataLoader(
    UNLABELED_DATASET,
    batch_size=batch_size_list[1],
    shuffle=True,
    num_workers=batch_size_list[1] // 2,
    worker_init_fn=my_worker_init_fn,
)
TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    worker_init_fn=my_worker_init_fn,
)

if FLAGS.use_wandb:
    wandb.init(project=("Diffusion-SS3D_train_" + str(FLAGS.data_ratio)), name=os.path.basename(LOG_DIR))
    wandb.config = {"learning_rate": BASE_LEARNING_RATE, "epochs": MAX_EPOCH, "batch_size": BATCH_SIZE}

# Used for Pseudo box generation and AP calculation
CONFIG_DICT = {
    "dataset_config": DATASET_CONFIG,
    "unlabeled_batch_size": batch_size_list[1],
    "dataset": FLAGS.dataset,
    "remove_empty_box": False,
    "use_3d_nms": True,
    "nms_iou": 0.25,
    "use_old_type_nms": False,
    "cls_nms": True,
    "use_iou_for_nms": FLAGS.use_iou_for_nms,
    "per_class_proposal": True,
    "conf_thresh": FLAGS.conf_thresh,
    "obj_threshold": 0.9,
    "cls_threshold": 0.9,
    "use_lhs": True,
    "iou_threshold": 0.25,
    "use_unlabeled_obj_loss": False,
    "use_unlabeled_vote_loss": False,
    "vote_loss_size_factor": 1.0,
    "samecls_match": False,
    "view_stats": FLAGS.view_stats,
}

DIFFUSION_CONFIG = {
    "dataset": FLAGS.dataset,
    "size_scale": FLAGS.size_snr_scale,
    "label_scale": FLAGS.label_snr_scale,
    "timesteps": FLAGS.timesteps,
    "sampling_timesteps": FLAGS.sampling_timesteps,
    "iterative_train": FLAGS.iterative_train,
    "renewal_obj": 0.9,
    "renewal_iou": 0.25,
    "renewal_sem_cls": 0.9,
    "size_mean_bias": 0.25,
    "center_mean_bias": 0.5,
    "label_mean_bias": 1 / 18,
    "label_loss_weight": 0.1,
}

for key in CONFIG_DICT.keys():
    if key != "dataset_config":
        log_string(key + ": " + str(CONFIG_DICT[key]))

for key in DIFFUSION_CONFIG.keys():
    log_string(key + ": " + str(DIFFUSION_CONFIG[key]))

def create_model(ema=False):
    model = VoteNet(
        num_class=DATASET_CONFIG.num_class,
        num_heading_bin=DATASET_CONFIG.num_heading_bin,
        num_size_cluster=DATASET_CONFIG.num_size_cluster,
        mean_size_arr=DATASET_CONFIG.mean_size_arr,
        dataset_config=DATASET_CONFIG,
        num_proposal=FLAGS.num_target,
        input_feature_dim=num_input_channel,
        vote_factor=FLAGS.vote_factor,
        diffusion_config=DIFFUSION_CONFIG,
    )
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector = create_model()
ema_detector = create_model(ema=True)

detector.to(device)
ema_detector.to(device)

train_labeled_criterion = get_labeled_loss
train_unlabeled_criterion = get_unlabeled_loss
test_detector_criterion = get_loss

# Load the Adam optimizer
optimizer = optim.Adam(detector.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
if FLAGS.detector_checkpoint is not None and os.path.isfile(FLAGS.detector_checkpoint):
    checkpoint = torch.load(FLAGS.detector_checkpoint)
    pretrained_dict = checkpoint["model_state_dict"]

    ########
    model_dict = detector.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    detector.load_state_dict(model_dict)
    model_dict = ema_detector.state_dict()
    model_dict.update(pretrained_dict)
    ema_detector.load_state_dict(model_dict)
    ########

    if FLAGS.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    epoch_ckpt = checkpoint["epoch"]
    print("Loaded checkpoint %s (epoch: %d)" % (FLAGS.detector_checkpoint, epoch_ckpt))


# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
# inherited this from VoteNet and SESS
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(detector, bn_lambda=bn_lbmd, last_epoch=-1)
if FLAGS.resume:
    bnm_scheduler.step(start_epoch)


def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

AUGMENT_HELPER = AugmentHelper(DATASET_CONFIG, mixup_ratio=FLAGS.mixup_ratio)

print("************************** GLOBAL CONFIG END **************************")
# ------------------------------------------------------------------------- GLOBAL CONFIG END


def train_one_epoch(global_step):
    stat_dict = {}  # collect statistics
    epoch_stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    detector.train()  # set model to training mode
    ema_detector.train()

    unlabeled_dataloader_iterator = iter(UNLABELED_DATALOADER)

    for batch_idx, batch_data_label in enumerate(LABELED_DATALOADER):
        try:
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)
        except StopIteration:
            unlabeled_dataloader_iterator = iter(UNLABELED_DATALOADER)
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)

        for key in batch_data_unlabeled:
            if type(batch_data_unlabeled[key]) == list:
                batch_data_label[key] = torch.cat([batch_data_label[key]] + batch_data_unlabeled[key], dim=0)
            else:
                batch_data_label[key] = torch.cat((batch_data_label[key], batch_data_unlabeled[key]), dim=0)

        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # inputs = {"point_clouds": batch_data_label["point_clouds"]}
        ema_inputs = {"point_clouds": batch_data_label["ema_point_clouds"]}

        optimizer.zero_grad()
        with torch.no_grad():
            ema_end_points = ema_detector.evaluate(ema_inputs, batch_data_label, jittering=True, ema=True)

        mixup_methods = [AUGMENT_HELPER.mix_up_1, AUGMENT_HELPER.mix_up_2, AUGMENT_HELPER.mix_up_3]
        mixup = mixup_methods[np.random.randint(3)]
        batch_data_label = mixup(batch_data_label, ema_end_points, CONFIG_DICT)
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        
        end_points_dict = detector(inputs, batch_data_label, ema_end_points=ema_end_points, jittering=True)

        total_loss = 0
        for iter_id in range(FLAGS.iterative_train):
            # Compute loss and gradients, update parameters.
            for key in batch_data_label:
                assert key not in end_points_dict[iter_id]
                end_points_dict[iter_id][key] = batch_data_label[key]

            detection_loss, end_points_dict[iter_id] = train_labeled_criterion(end_points_dict[iter_id], DATASET_CONFIG, DIFFUSION_CONFIG)

            unlabeled_loss, end_points_dict[iter_id] = train_unlabeled_criterion(end_points_dict[iter_id], ema_end_points, DATASET_CONFIG, CONFIG_DICT, DIFFUSION_CONFIG)

            loss = detection_loss + unlabeled_loss * FLAGS.unlabeled_loss_weight

            end_points_dict[iter_id]["loss"] = loss

            total_loss = total_loss + loss

        total_loss.backward()

        optimizer.step()
        global_step += 1
        update_ema_variables(detector, ema_detector, FLAGS.ema_decay, global_step)

        iter_id = FLAGS.iterative_train - 1
        # Accumulate statistics and print out
        for key in end_points_dict[iter_id]:
            if "loss" in key or "acc" in key or "ratio" in key or "value" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points_dict[iter_id][key].item()
                if key not in epoch_stat_dict:
                    epoch_stat_dict[key] = 0
                epoch_stat_dict[key] += end_points_dict[iter_id][key].item()

        batch_interval = FLAGS.print_interval
        if (batch_idx + 1) % batch_interval == 0:
            log_string(" ---- batch: %03d ----" % (batch_idx + 1))

            for key in sorted(stat_dict.keys()):
                log_string("mean %s: %f" % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0

    for key in epoch_stat_dict.keys():
        epoch_stat_dict[key] /= len(LABELED_DATALOADER)

    return global_step, epoch_stat_dict


AP_IOU_THRESHOLDS = [0.25, 0.5]
BEST_MAP = [0.0, 0.0]


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    ap_calculator_list = [
        APCalculator(iou_thresh, DATASET_CONFIG.class2type)
        for iou_thresh in AP_IOU_THRESHOLDS
    ]
    detector.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {"point_clouds": batch_data_label["point_clouds"]}
        with torch.no_grad():
            end_points = detector.evaluate(inputs, batch_data_label, jittering=False, ema=False)

        # Compute loss
        for key in batch_data_label:
            assert key not in end_points
            end_points[key] = batch_data_label[key]
        _, end_points = test_detector_criterion(end_points, DATASET_CONFIG, DIFFUSION_CONFIG, EPOCH_CNT)

        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)

        # Accumulate statistics and print out
        for key in end_points:
            if "loss" in key or "acc" in key or "ratio" in key or "value" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string("eval mean %s: %f" % (key, stat_dict[key] / (float(batch_idx + 1))))

    # Evaluate average precision
    map = []
    for i, ap_calculator in enumerate(ap_calculator_list):
        print("-" * 10, "iou_thresh: %f" % (AP_IOU_THRESHOLDS[i]), "-" * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string("eval %s: %f" % (key, metrics_dict[key]))

        map.append(metrics_dict["mAP"])

    mean_loss = stat_dict["detection_loss"] / float(batch_idx + 1)

    for key in sorted(stat_dict.keys()):
        stat_dict[key] /= len(TEST_DATALOADER)

    return mean_loss, map, stat_dict


def evaluate_with_opt():
    stat_dict = {}  # collect statistics
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    detector.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {"point_clouds": batch_data_label["point_clouds"]}

        optimizer.zero_grad()
        end_points = detector.evaluate(inputs, batch_data_label, jittering=False, ema=False, iou_opt=True)
        center = end_points["center"]
        size_class = torch.argmax(end_points["size_scores"], dim=-1)
        sem_cls = end_points["sem_cls_scores"].argmax(-1)
        size = end_points["size"]
        heading = end_points["heading"]
        iou = end_points["iou_scores"]
        origin_iou = iou.clone()
        iou_gathered = torch.gather(iou, dim=2, index=sem_cls.unsqueeze(-1).detach()) .squeeze(-1) .contiguous() .view(-1)
        iou_gathered.backward(torch.ones(iou_gathered.shape).cuda())
        # max_iou = iou_gathered.view(center.shape[:2])
        center_grad = center.grad
        size_grad = size.grad
        mask = torch.ones(center.shape).cuda()
        count = 0

        for k in end_points.keys():
            end_points[k] = end_points[k].detach()
        while True:
            center_ = center.detach() + FLAGS.opt_rate * center_grad * mask
            size_ = size.detach() + FLAGS.opt_rate * size_grad * mask
            heading_ = heading.detach()
            optimizer.zero_grad()
            center_.requires_grad = True
            size_.requires_grad = True
            end_points_ = detector.forward_onlyiou_faster(end_points, center_, size_, heading_)
            iou = end_points_["iou_scores"]
            iou_gathered = torch.gather(iou, dim=2, index=sem_cls.unsqueeze(-1).detach()) .squeeze(-1) .contiguous() .view(-1)
            iou_gathered.backward(torch.ones(iou_gathered.shape).cuda())
            center_grad = center_.grad
            size_grad = size_.grad
            # cur_iou = iou_gathered.view(center.shape[:2])
            # mask[cur_iou < max_iou - 0.1] = 0
            # mask[torch.abs(cur_iou - max_iou) < 0.001] = 0
            # print(mask.sum().float() /mask.view(-1).shape[-1])
            count += 1
            if count > FLAGS.opt_step:
                break
            center = center_
            size = size_

        end_points["center"] = center_
        B, K = size_class.shape[:2]
        mean_size_arr = DATASET_CONFIG.mean_size_arr
        mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster,3)
        size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
        size_base = size_base.view(B, K, 3)
        end_points["size_residuals"] = ((size_ * 2 - size_base).unsqueeze(2).expand(-1, -1, DATASET_CONFIG.num_size_cluster, -1))
        optimizer.zero_grad()

        # Compute loss
        for key in batch_data_label:
            assert key not in end_points
            end_points[key] = batch_data_label[key]
            
        loss, end_points = test_detector_criterion(end_points, DATASET_CONFIG, DIFFUSION_CONFIG, EPOCH_CNT)

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)

        # Accumulate statistics and print out
        for key in end_points:
            if "loss" in key or "acc" in key or "ratio" in key or "value" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        # ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string("eval mean %s: %f" % (key, stat_dict[key] / (float(batch_idx + 1))))

    # Evaluate average precision
    map = []
    for i, ap_calculator in enumerate(ap_calculator_list):
        print("-" * 10, "iou_thresh: %f" % (AP_IOU_THRESHOLDS[i]), "-" * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string("eval %s: %f" % (key, metrics_dict[key]))

        map.append(metrics_dict["mAP"])

    mean_loss = stat_dict["detection_loss"] / float(batch_idx + 1)
    
    return mean_loss, map


def wandb_log(epoch, train_stat_dict, eval_stat_dict, map, best_map):

    to_log = {}

    to_log["epoch"] = epoch
    to_log["learning rate"] = get_current_lr(epoch)

    for key, value in train_stat_dict.items():
        to_log["train_" + key] = value

    if eval_stat_dict != None:
        for key, value in eval_stat_dict.items():
            to_log["eval_" + key] = value

        to_log["mAP 0.25"] = map[0]
        to_log["mAP 0.5"] = map[1]

    to_log["Best mAP 0.25"] = best_map[0]
    to_log["Best mAP 0.5"] = best_map[1]
    to_log["Best mAP (0.25 + 0.5)"] = best_map[0] + best_map[1]

    wandb.log(to_log)


def train():
    global EPOCH_CNT
    global start_epoch
    global_step = 0
    loss = 0
    EPOCH_CNT = 0
    global BEST_MAP
    if FLAGS.eval:
        np.random.seed()
        if FLAGS.opt_step > 0:
            evaluate_with_opt()
        else:
            evaluate_one_epoch()
        sys.exit(0)
    start_from = 0
    if FLAGS.resume:
        start_from = start_epoch
    for epoch in range(start_from, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string("\n**** EPOCH %03d, STEP %d ****" % (epoch, global_step))
        log_string("Current epoch: %d, obj threshold = %.3f & cls threshold = %.3f"% (epoch, CONFIG_DICT["obj_threshold"], CONFIG_DICT["cls_threshold"]))
        log_string("Current learning rate: %f" % (get_current_lr(epoch)))
        log_string("Current BN decay momentum: %f" % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))

        # in numpy 1.18.5 this actually sets `np.random.get_state()[1][0]` to default value
        # so the test data is consistent as the initial seed is the same
        np.random.seed()
        global_step, train_stat_dict = train_one_epoch(global_step)
        
        map = [0.0, 0.0]
        eval_stat_dict = None

        if EPOCH_CNT > 0 and EPOCH_CNT % FLAGS.eval_interval == 0:
            loss, map, eval_stat_dict = evaluate_one_epoch()

            if map[0] + map[1] > BEST_MAP[0] + BEST_MAP[1]:
                BEST_MAP = map
                save_dict = {
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                }
                try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                    save_dict["model_state_dict"] = detector.module.state_dict()
                    save_dict["ema_model_state_dict"] = ema_detector.module.state_dict()
                except:
                    save_dict["model_state_dict"] = detector.state_dict()
                    save_dict["ema_model_state_dict"] = ema_detector.state_dict()
                torch.save(save_dict, os.path.join(LOG_DIR, "best_checkpoint_sum.tar"))
            PERFORMANCE_FOUT.write(
                "epoch: "
                + str(EPOCH_CNT)
                + "\n"
                + "best: "
                + str(BEST_MAP[0].item())
                + ", "
                + str(BEST_MAP[1].item())
                + "\n"
            )
            PERFORMANCE_FOUT.flush()

        if FLAGS.use_wandb:
            wandb_log(epoch, train_stat_dict, eval_stat_dict, map, BEST_MAP)


if __name__ == "__main__":
    train()
