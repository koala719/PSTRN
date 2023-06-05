from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1, 20)

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
from dali_dataset.preprocess_fashion_mnist import get_fashion_mnist_iter_dali
from config import proj_cfg

from model.lenet import LeNet5_Mnist
from model.lenet_tensor import LeNet5_Mnist_Tensor
from model.lenet_tensor_dropout import LeNet5_Mnist_Tensor_Dropout
from model.lenet_tensor_nannan import LeNet5_Fashion_Mnist_Tensor
# from model.lenet import LeNet5_Fashion_Mnist


Models = dict(
    LeNet5_Mnist=LeNet5_Mnist,
    LeNet5_Mnist_Tensor=LeNet5_Mnist_Tensor,
    LeNet5_Mnist_Tensor_Dropout=LeNet5_Mnist_Tensor_Dropout,
    LeNet5_Fashion_Mnist_Tensor=LeNet5_Fashion_Mnist_Tensor,
)


def main(rank_code, no_cuda=False, seed=233, lr=2e-3, start_epoch=0, weight_conv_dir=None,
         pre_train_model_path=None, dataset='./datasets', batch_size=512, dali_cpu=False, num_threads=4,
         epochs=20, save_freq=0, continual_save_freq=3, model=LeNet5_Fashion_Mnist_Tensor,save=None):
    # ---------------------- Set GPU ----------------------

    use_cuda = not no_cuda and torch.cuda.is_available()
    print('---------------', torch.cuda.is_available())

    gpu_num = torch.cuda.device_count()
    print('---------------------------', gpu_num)
    multi_gpus = gpu_num > 1
    device = torch.device("cuda" if use_cuda else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # ------------------- Define Model ------------------------

    rank_code = rank_code.tolist()
    RANK = []
    for i in range(7):
        rank = rank_code[i*5] * (2**4) + rank_code[i*5+1] * (2**3) + \
               rank_code[i*5+2] * (2**2) + rank_code[i*5+3] * (2**1) + rank_code[i*5+4] * (2**0)
        if rank < 2:
            rank = 2
        RANK[i].append(rank)

    print ("---", rank_code, "---")
    # if model in Models:
    #     model = Models[model](num_classes=10, RANK=rank_code)
    # else:
    #     raise ValueError("model %s is not existed!" % model)
    model = LeNet5_Fashion_Mnist_Tensor(num_classes=10, RANK=RANK)


    criterion = nn.CrossEntropyLoss()

    # for param in model.model1.parameters():
    #     param.requires_grad = False
    #
    # param_fc = [model.fc5.parameters(), model.fc6.parameters()]
    #
    # optimizer = optim.Adam(
    #     param_fc, lr=lr,
    # )

    optimizer = optim.Adam(
        model.parameters(), lr=lr,
    )
    lr_schedule = MultiStepLR(optimizer, [5, 10, 15])

    model.to(device)

    # ----------------- Continual Training -------------------------
    # Load Model, Optimizer
    # Set model_dir, log_dir, and start_epoch

    best_top_1 = - math.inf
    start_epoch = start_epoch

    if weight_conv_dir:
        weight_conv_info = ContinualTrain.read(weight_conv_dir)
        model_dir, log_dir, model_path, optimizer_path, lr_schedule_path, best_top_1 = weight_conv_info
        print("Loading the Model1 ...")
        model.model1 = load_model(multi_gpus, model.model1, model_path)


    # if continual_train_model_dir:
    #     continual_info = ContinualTrain.read(continual_train_model_dir)
    #     model_dir, log_dir, model_path, optimizer_path, lr_schedule_path, best_top_1 = continual_info
    #
    #     optimizer.load_state_dict(torch.load(optimizer_path))
    #     lr_schedule.load_state_dict(torch.load(lr_schedule_path))
    #
    #     print("Loading the Model of Epoch %d..." % lr_schedule.last_epoch)
    #     model = load_model(multi_gpus, model, model_path)
    #
    #     start_epoch = lr_schedule.last_epoch

    # else:

        # ------------------- Set Model-Save Path ----------------------

    model_root = os.path.join(proj_cfg.save_root, "mnist", save,
                              time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "_%d" % seed)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # ------------------- Save Running Setting ----------------------

    # with open(os.path.join(model_root, "running_setting.txt"), "w") as f:
    #     f.write(str(args))

    # ------------------- Set Models-Save Path ----------------------

    model_dir = os.path.join(model_root, "checkpoint")
    create_nonexistent_folder(model_dir)

    # ------------------- Set Logs-Save Path ----------------------

    log_dir = os.path.join(model_root, "logs")
    create_nonexistent_folder(log_dir)

    if pre_train_model_path:
        print("Loading Pre-trained Model...")
        model = load_model(multi_gpus, model, pre_train_model_path)

    # ------------- Assign Model to GPUS ----------------------

    if multi_gpus:
        print("DataParallel...")
        model = nn.DataParallel(model)

    # --------------------- Set DataLoader -----------------

    train_loader = get_fashion_mnist_iter_dali(
        type='train', image_dir=dataset, batch_size=batch_size, dali_cpu=dali_cpu,
        num_threads=num_threads, seed=seed, gpu_num=gpu_num,
    )
    test_loader = get_fashion_mnist_iter_dali(
        type='val', image_dir=dataset, batch_size=batch_size, gpu_num=gpu_num,
        num_threads=num_threads, seed=seed, dali_cpu=dali_cpu,
    )

    # ------------------- Set Summary Writer ----------------------

    summary_writer = SummaryWriter(logdir=log_dir)

    # --------------- Training --------------

    save_top_1_model_flag = False
    for epoch in range(start_epoch, epochs):
        summary_writer.add_scalar(r"learning_rate", optimizer.param_groups[0]['lr'], epoch)

        print("\nTraining Epoch: %d/%d" % (epoch, epochs - 1))
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
        if save_freq:
            if epoch % save_freq == 0:
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
        if continual_save_freq:
            if epoch % continual_save_freq == 0:
                summary_writer.flush()
                ContinualTrain.save(model_dir, log_dir, model, optimizer, lr_schedule, best_top_1)
    compression_rat = model.compression_rat
    summary_writer.close()
    print("Run Over!")

    print ('valid_acc:', best_top_1*100, 'compression_ratio:', compression_rat)

    return {'valid_acc': best_top_1*100, 'compression_ratio': compression_rat}

