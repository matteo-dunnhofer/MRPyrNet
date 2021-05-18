import sys
sys.path.append('..')

import os
import time
from datetime import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from mrnetdataset import MRNetDataset
from mrpyrnet import MRPyrNet
from sklearn import metrics
import csv
import modules.utils as ut
import math


def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, device, log_every=100):
    model.train()

    model = model.to(device)

    y_preds = []
    y_preds_mcc = []
    y_trues = []
    losses = []
    auc = float(0)

    for i, (image, label) in enumerate(train_loader):

        image = image.to(device)
        label = label.to(device)

        prediction1, prediction2, prediction3, prediction4, prediction5  = model(image) 

        loss = F.cross_entropy(prediction1, label[0]) + \
                F.cross_entropy(prediction2, label[0]) + \
                F.cross_entropy(prediction3, label[0]) + \
                F.cross_entropy(prediction4, label[0]) + \
                F.cross_entropy(prediction5, label[0])


        optimizer.zero_grad()      
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas1 = F.softmax(prediction1, dim=1)
        probas2 = F.softmax(prediction2, dim=1)
        probas3 = F.softmax(prediction3, dim=1)
        probas4 = F.softmax(prediction4, dim=1)
        probas5 = F.softmax(prediction5, dim=1)

        y_trues.append(int(label[0, 0]))
        probas = torch.cat([probas1,probas2,probas3,probas4,probas5], dim=0)
        
        y_preds.append(probas[:, 1].max().float().item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value, epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6} '''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  ))

    writer.add_scalar('Train/AUC_epoch', auc, epoch)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)

    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, device, log_every=20):
    model.eval()

    y_trues = []
    y_preds = []
    y_class_preds = []
    losses = []
    auc = float(0)

    for i, (image, label) in enumerate(val_loader):

        image = image.to(device)
        label = label.to(device)

        prediction1, prediction2, prediction3, prediction4, prediction5  = model(image) 

        loss = F.cross_entropy(prediction1, label[0]) + \
                F.cross_entropy(prediction2, label[0]) + \
                F.cross_entropy(prediction3, label[0]) + \
                F.cross_entropy(prediction4, label[0]) + \
                F.cross_entropy(prediction5, label[0])

        loss_value = loss.item()
        losses.append(loss_value)

        probas1 = F.softmax(prediction1, dim=1)
        probas2 = F.softmax(prediction2, dim=1)
        probas3 = F.softmax(prediction3, dim=1)
        probas4 = F.softmax(prediction4, dim=1)
        probas5 = F.softmax(prediction5, dim=1)

        y_trues.append(int(label[0, 0]))
        probas = torch.cat([probas1,probas2,probas3,probas4,probas5], dim=0)
  
        y_preds.append(probas[:, 1].max().float().item())
        y_class_preds.append((probas[:, 1].max() > 0.5).float().item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except Exception as e:
            auc = 0.5

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)


        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss: {4} | val auc : {5} | lr : {6} '''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Val/AUC_epoch', auc, epoch)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_trues, y_class_preds)
    val_accuracy = np.round(val_accuracy, 4)
    val_sensitivity = np.round(val_sensitivity, 4)
    val_specificity = np.round(val_specificity, 4)

    return val_loss_epoch, val_auc_epoch, val_accuracy, val_sensitivity, val_specificity

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create dirs to store experiment checkpoints, logs, and results
    exp_dir_name = args.experiment
    exp_dir = os.path.join('experiments', exp_dir_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, 'models'))
        os.makedirs(os.path.join(exp_dir, 'logs'))
        os.makedirs(os.path.join(exp_dir, 'results'))

    log_root_folder = exp_dir + "/logs/{0}/{1}/".format(args.task, args.plane)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    ##-----SAMPLER------
    
    # create training and validation set
    train_dataset = MRNetDataset(args.path_to_data, args.task, args.plane, train=True)
    train_dataset_weights = [train_dataset.weights[train_dataset.labels[i]] for i in range(len(train_dataset))]
    train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=4, drop_last=False)
    
    validation_dataset = MRNetDataset(args.path_to_data, args.task, args.plane, train=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MRPyrNet(K=args.K, norm_type=args.set_norm_type, seed=args.seed, D=args.D)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    best_val_loss = float('inf')
    best_val_auc = float(0)
    best_val_accuracy = float(0)
    best_val_sensitivity = float(0)
    best_val_specificity = float(0)

    num_epochs = args.epochs
    log_every = args.log_every

    t_start_training = time.time()
    
    # train and test loop
    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        t_start = time.time()
        
        # train
        train_loss, train_auc = train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, device, log_every)
        
        # evaluate
        val_loss, val_auc, val_accuracy, val_sensitivity, val_specificity = evaluate_model(model, validation_loader, epoch, num_epochs, writer, current_lr, device)

        t_end = time.time()
        delta = t_end - t_start

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_val_accuracy = val_accuracy
            best_val_sensitivity = val_sensitivity
            best_val_specificity = val_specificity
            if bool(args.save_model):
                file_name = f'model_{args.prefix_name}_{args.task}_{args.plane}.pth'
                for f in os.listdir(exp_dir + '/models/'):
                    if (args.task in f) and (args.plane in f) and (args.prefix_name in f):
                        os.remove(exp_dir + f'/models/{f}')
                torch.save(model, exp_dir + f'/models/{file_name}')

    # save results to csv file
    with open(os.path.join(exp_dir, 'results', f'model_{args.prefix_name}_{args.task}_{args.plane}-results.csv'), 'w') as res_file:
        fw = csv.writer(res_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(['LOSS', 'AUC-best', 'Accuracy-best', 'Sensitivity-best', 'Specificity-best'])
        fw.writerow([best_val_loss, best_val_auc, best_val_accuracy, best_val_sensitivity, best_val_specificity])
        res_file.close()

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--D', type=int, default=5)
    parser.add_argument('--set_norm_type', type=str, choices=['instance', 'layer'], default='layer')
    parser.add_argument('--K', type=int, choices=[1,2,3,4], default=4)
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    run(args)
