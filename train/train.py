# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020-02-10 09:59

import sys
# update your projecty root path before running
sys.path.insert(0, '/home/lnn/tensor-net-master')

from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(3, 20)

import os
import time
import argparse

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm

from utils.util import load_model, eval_model_by_dali, ContinualTrain, create_nonexistent_folder
from utils.metric import MetricAccuracy, MetricLoss
# from dali_dataset import get_mnist_iter_dali
from dali_dataset.preprocess_fashion_mnist import get_fashion_mnist_iter_dali
from search.config import proj_cfg

from model.lenet import LeNet5_Mnist
from model.lenet_tensor import LeNet5_Mnist_Tensor
from model.lenet_tensor_dropout import LeNet5_Mnist_Tensor_Dropout
from model.lenet_tensor_fc_ours import LeNet5_Mnist_Tensor_NANNAN_OURS
from model.lenet_tensor_fc_ours import LeNet5_Fashion_Mnist_Tensor_OURS


Models = dict(
    LeNet5_Mnist=LeNet5_Mnist,
    LeNet5_Mnist_Tensor=LeNet5_Mnist_Tensor,
    LeNet5_Mnist_Tensor_Dropout=LeNet5_Mnist_Tensor_Dropout,
    LeNet5_Mnist_Tensor_NANNAN_OURS=LeNet5_Mnist_Tensor_NANNAN_OURS,
    LeNet5_Fashion_Mnist_Tensor_OURS=LeNet5_Fashion_Mnist_Tensor_OURS,
)


def main(args):

    # ---------------------- Set GPU ----------------------

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_num = torch.cuda.device_count()
    multi_gpus = gpu_num > 1
    device = torch.device("cuda" if use_cuda else "cpu")

    random.seed(args.random_seed)
    np.random.seed(args.torch_seed)
    torch.manual_seed(args.torch_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.torch_seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # ------------------- Define Model ------------------------
    if args.model in Models:
        model = Models[args.model](num_classes=10)
    else:
        raise ValueError("model %s is not existed!" % args.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr,
    )
    lr_schedule = MultiStepLR(optimizer, [150, 250])

    model.to(device)

    # ----------------- Continual Training -------------------------
    # Load Model, Optimizer
    # Set model_dir, log_dir, and start_epoch

    best_top_1 = - math.inf
    start_epoch = args.start_epoch

    if args.continual_train_model_dir:
        continual_info = ContinualTrain.read(args.continual_train_model_dir)
        model_dir, log_dir, model_path, optimizer_path, lr_schedule_path, best_top_1 = continual_info

        optimizer.load_state_dict(torch.load(optimizer_path))
        lr_schedule.load_state_dict(torch.load(lr_schedule_path))

        print("Loading the Model of Epoch %d..." % lr_schedule.last_epoch)
        model = load_model(multi_gpus, model, model_path)

        start_epoch = lr_schedule.last_epoch

    else:

        # ------------------- Set Model-Save Path ----------------------

        model_root = os.path.join(proj_cfg.save_root, "mnist", args.model,
                                  time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        if not os.path.exists(model_root):
            os.makedirs(model_root)

        # ------------------- Save Running Setting ----------------------

        with open(os.path.join(model_root, "running_setting.txt"), "w") as f:
            f.write(str(args))

        # ------------------- Set Models-Save Path ----------------------

        model_dir = os.path.join(model_root, "checkpoint")
        create_nonexistent_folder(model_dir)

        # ------------------- Set Logs-Save Path ----------------------

        log_dir = os.path.join(model_root, "logs")
        create_nonexistent_folder(log_dir)

        if args.pre_train_model_path:
            print("Loading Pre-trained Model...")
            model = load_model(multi_gpus, model, args.pre_train_model_path)

        # ------------- Assign Model to GPUS ----------------------

        if multi_gpus:
            print("DataParallel...")
            model = nn.DataParallel(model)

    # --------------------- Set DataLoader -----------------

    train_loader = get_fashion_mnist_iter_dali(
        type='train', image_dir=args.dataset, batch_size=args.train_batch_size, dali_cpu=args.dali_cpu,
        num_threads=args.num_threads, seed=args.pipeline_seed, gpu_num=gpu_num,
    )
    test_loader = get_fashion_mnist_iter_dali(
        type='val', image_dir=args.dataset, batch_size=args.test_batch_size, gpu_num=gpu_num,
        num_threads=args.num_threads, seed=args.pipeline_seed, dali_cpu=args.dali_cpu,
    )

    # ------------------- Set Summary Writer ----------------------

    summary_writer = SummaryWriter(logdir=log_dir)

    # --------------- Training --------------

    save_top_1_model_flag = False
    for epoch in range(start_epoch, args.epochs):
        summary_writer.add_scalar(r"learning_rate", optimizer.param_groups[0]['lr'], epoch)

        print("\nTraining Epoch: %d/%d" % (epoch, args.epochs-1))
        model.train()

        start_time_epoch = time.time()
        with tqdm(total=len(train_loader)) as pbar:
                for data_pairs in train_loader:
                    for data_pair in data_pairs:
                        data = data_pair["data"].to(device, non_blocking=True)
                        target = data_pair["label"].squeeze().long().to(device, non_blocking=True).view(-1)

                        optimizer.zero_grad()

                        output_logits = model(data)

                        loss = criterion(output_logits, target)

                        loss.backward()
                        optimizer.step()

                        pbar.update(1)

        # Update Learning Rate
        lr_schedule.step()

        end_time_epoch = time.time()
        train_time_epoch = end_time_epoch - start_time_epoch
        summary_writer.add_scalar("epoch_train_time", train_time_epoch, epoch)

        # Derive the Results of Training and Testing
        training_metrics = [MetricAccuracy(1), MetricLoss(criterion)]
        testing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss(criterion)]

        print("Evaluating Training Data...")
        training_metrics = eval_model_by_dali(model, device, train_loader, training_metrics, gpu_num=gpu_num)

        res_training = []
        for training_metric in training_metrics:
            res_training.append(training_metric.get_metric())

        print("Evaluating Testing Data...")
        testing_metrics = eval_model_by_dali(model, device, test_loader, testing_metrics, gpu_num=gpu_num)

        res_test = []
        for metric in testing_metrics:
            res_test.append(metric.get_metric())

        # save logs
        for train_metric in res_training:
            summary_writer.add_scalar(r"train/" + train_metric["metric_name"], train_metric["metric_value"], epoch)

        for test_metric in res_test:
            summary_writer.add_scalar(r"test/" + test_metric["metric_name"], test_metric["metric_value"], epoch)

            # Update Test Top_1
            if test_metric["metric_name"] == "top_1":
                if test_metric["metric_value"] > best_top_1:
                    print("Accuracy(%f) of Epoch %d is best now." % (test_metric["metric_value"], epoch))

                    best_top_1 = test_metric["metric_value"]

                    summary_writer.add_scalar(r"test_best_top1_acc", best_top_1, epoch)
                    save_top_1_model_flag = True

        # save checkpoint
        if args.save_freq:
            if epoch % args.save_freq == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
                torch.save(optimizer.state_dict(),
                           os.path.join(model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))
        else:
            if save_top_1_model_flag:
                print("Saving Best Model...")
                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'model-nn-best.pt'))
                torch.save(optimizer.state_dict(),
                           os.path.join(model_dir, 'opt-nn-checkpoint-best.tar'))
                torch.save(lr_schedule.state_dict(),
                           os.path.join(model_dir, 'lr-sdl-nn-checkpoint-best.tar'))

                save_top_1_model_flag = False

        # save continual training model
        if args.continual_save_freq:
            if epoch % args.continual_save_freq == 0:
                summary_writer.flush()
                ContinualTrain.save(model_dir, log_dir, model, optimizer, lr_schedule, best_top_1)

    summary_writer.close()
    print("Run Over!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on MNIST")

    parser.add_argument("--dataset", type=str, default='./datasets/')
    parser.add_argument("--model", type=str, default="LeNet5_Mnist_Tensor_NANNAN_OURS")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')

    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument('--pre-train-model-path', type=str, default=None,
                        help='a pre-trained model path, set to None to disable the pre-training')
    parser.add_argument('--continual-train-model-dir', type=str, default=None,
                        help='continual training model folder, set to None to disable the keep training')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dali-cpu', action='store_true', default=False, help='disables CUDA dali')

    parser.add_argument('--save-freq', type=int, default=0, help='save frequency, set 0 to activate save best')
    parser.add_argument('--continual-save-freq', type=int, default=3, help='continual save frequency, set 0 to close')

    parser.add_argument('--random-seed', type=int, default=233, help='normal random seed')
    parser.add_argument('--torch-seed', type=int, default=233, help='torch random seed')
    parser.add_argument('--pipeline-seed', type=int, default=233, help='dali pipeline random seed')

    args = parser.parse_args()
    print(args)

    main(args)
