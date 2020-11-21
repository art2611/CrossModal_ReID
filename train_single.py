import torch
import argparse
import os
import time
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from data_loader import *
from torchvision import transforms
from utils import *
from loss import BatchHardTripLoss
from tensorboardX import SummaryWriter
from model import Network, Network_fuse
from multiprocessing import freeze_support
from data_augmentation import data_aug
import numpy as np
import matplotlib.pyplot as plt
from evaluation import eval_regdb
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def multi_process() :
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
    parser.add_argument('--train', default='visible', help='train visible or thermal only')
    args = parser.parse_args()

    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter("runs/ThermalOnly1")

    # Init variables :
    img_w = 144
    img_h = 288
    test_batch_size = 64
    batch_num_identities = 8 # 8 different identities in a batch
    batch_num_identities = 16 # 16 different identities in a batch
    num_of_same_id_in_batch = 4 # Number of same identity in a batch
    workers = 4
    lr = 0.001
    checkpoint_path = '../save_model/'
    if args.train == 'visible' :
        suffix = f'RegDB_person_Visible_only({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
    elif args.train == "thermal" :
        suffix = f'RegDB_person_Thermal_only({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

    # Data info  :
    data_path = '../Datasets/RegDB/'
    #log_path = args.log_path + 'regdb_log/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((img_h, img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    # print(f'Image loaded : {len(np.unique(trainset.train_color_label))}')
    # print(f'len(img valid) {len(np.unique(validset.valid_color_label))}')
    # print(f'len(img train) {len(np.unique(testset.test_color_label))}')

    # generate the idx of each person identity for instance, identity 10 have the index 100 to 109
    # It is a list of list train_color_pos[10] = [100, ..., 109]
    # train_color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)


    ######################################### TRAIN & VALIDATION SETs
    Timer1 = time.time()
    print('==> Loading images..')
    dataset = 'sysu'
    # dataset = 'regdb'
    if dataset == 'sysu':
        print('==> Trainset ..')
        # training set
        trainset = SYSUData_split(data_path, transform=transform_train, split = "training")
        # generate the idx of each person identity

        print('==> Validset ..')
        validset = SYSUData_split(data_path, transform=transform_train, split = "validation")
        print("==> Loaded")
        # print(f'Nombres d\'ids train color: {len(np.unique(trainset.train_color_label))}')
        # print(f'Nombres d\'ids valid color: {len(np.unique(validset.valid_color_label))}')
        # print(f'Nombres d\'ids train thermal : {len(np.unique(trainset.train_color_label))}')
        # print(f'Nombres d\'ids valid thermal : {len(np.unique(validset.valid_thermal_label))}')
        train_color_pos, train_thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        valid_color_pos, valid_thermal_pos = GenIdx(validset.valid_color_label, validset.valid_thermal_label)

        # print(f'Nombres d\'images en 0 train color: {len(train_color_pos[2])}')
        # print(f'Nombres d\'images en 0 valid color: {len(valid_color_pos[2])}')
        # print(f'Nombres d\'images en 0 train thermal : {len(train_thermal_pos[0])}')
        # print(f'Nombres d\'images en 0 valid thermal : {len(valid_thermal_pos[0])}')
    if dataset == "regdb" :
        if args.train == "visible ":
            trainset = RegDBVisibleData(data_path, transform=transform_train, split="training")
            validset = RegDBVisibleData(data_path, transform=transform_train, split="validation")
            print(f'Loaded images : {len(trainset.train_color_image) + len(validset.valid_color_label)}')
        elif args.train == "thermal" :
            trainset = RegDBThermalData(data_path, transform=transform_train, split="training")
            validset = RegDBThermalData(data_path, transform=transform_train, split="validation")
            print(f'Loaded images : {len(trainset.train_thermal_image) + len(validset.valid_thermal_label)}')

    # print(f'len(trainset.train_color_label) : {len(trainset.train_color_label)}')
    # print(f'len(validset.valid_color_label) : {len(validset.valid_color_label)}')

    print(' ')
    ######################################### Image GENERATION

    print('==> Image generation..')
    if dataset == 'regdb' :
        if args.train == "visible" :
            trainset.train_color_image, trainset.train_color_label, _, _ =\
                data_aug(visible_images = trainset.train_color_image, Visible_labels = trainset.train_color_label)
            validset.valid_color_image, validset.valid_color_label, _, _ =\
                data_aug(visible_images = validset.valid_color_image, Visible_labels = validset.valid_color_label)
            train_color_pos, _ = GenIdx(trainset.train_color_label, trainset.train_color_label)
            valid_color_pos, _ = GenIdx(validset.valid_color_label, validset.valid_color_label)

            print(f'New image number : {len(trainset.train_color_image)+ len(validset.valid_color_image)}')
        elif args.train == "thermal" :
            _, _, trainset.train_thermal_image, trainset.train_thermal_label =\
                data_aug(Thermal_images = trainset.train_thermal_image, Thermal_labels = trainset.train_thermal_label)
            _, _, validset.valid_thermal_image, validset.valid_thermal_label =\
                data_aug(Thermal_images = validset.valid_thermal_image, Thermal_labels = validset.valid_thermal_label)
            train_thermal_pos, _ = GenIdx(trainset.train_thermal_label, trainset.train_thermal_label)
            valid_thermal_pos, _ = GenIdx(validset.valid_thermal_label, validset.valid_thermal_label)

            print(f'New image number : {len(trainset.train_thermal_image) + len(validset.valid_thermal_image)}')

    ######################################### IMAGE DISPLAY
    for i in range(6):
         plt.subplot(2, 3, i + 1)
         plt.imshow(trainset.train_color_label[i])
         plt.imshow(trainset.train_color_label[i], cmap='gray')
    plt.show()
    sys.exit()
    ######################################### DATASET PROPERTIES
    # print(len(valid_color_pos[0]))
    # print(len(train_color_pos[0]))
    if args.train == "visible":

        print(f'Identities number : {len(train_color_pos)}')
        print(' ')
        print('New dataset statistics:')
        print('   set     |  Nb ids |  Nb img    ')
        print('  ------------------------------')
        print(f'  train_Visible  | {len(np.unique(trainset.train_color_label)):5d} | {len(trainset.train_color_label):8d}')
        print(f'  valid_Visible  | {len(np.unique(validset.valid_color_label)):5d} | {len(validset.valid_color_label):8d}')
        print('  ------------------------------')
        class_number = len(np.unique(trainset.train_color_label))
    elif args.train == "thermal":

        print(f'Identities number : {len(train_thermal_pos)}')
        print(' ')
        print('New dataset statistics:')
        print('   set     |  Nb ids |  Nb img    ')
        print('  ------------------------------')
        print(
            f'  train_Thermal  | {len(np.unique(trainset.train_thermal_label)):5d} | {len(trainset.train_thermal_label):8d}')
        print(
            f'  valid_Thermal  | {len(np.unique(validset.valid_thermal_label)):5d} | {len(validset.valid_thermal_label):8d}')
        class_number = len(np.unique(trainset.train_thermal_label))

    print(f'Data Loading Time:\t {time.time() - Timer1:.3f}')
    print(' ')
    print('==> Building model..')
    ######################################### MODEL TRAIN

    net = Network(class_number).to(device)

    ######################################### TRAINING
    print('==> Start Training...')

    #Train function
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.fc.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr},
        {'params': net.bottleneck.parameters(), 'lr': lr},
        {'params': net.fc.parameters(), 'lr': lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    ################FUNCTIONs :


    def training(epoch):
        current_lr = adjust_learning_rate(optimizer, epoch, lr=lr)
        train_loss = AverageMeter()
        id_loss = AverageMeter()
        tri_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0

        # switch to train mode
        net.train()
        end = time.time()
        for batch_idx, (input, label) in enumerate(trainloader):
            input = Variable(input.cuda())
            label = Variable(label.cuda())
            # input = Variable(input)
            # label = Variable(label)

            data_time.update(time.time() - end)

            # feat is the feature vector out of
            # Out is the last output
            feat, out0, = net(input)  # Call the visible branch only
            # feat, out0, = net(input, input, modal=1)  # Call the visible branch only

            loss_ce = criterion_id(out0, label)
            loss_tri, batch_acc = criterion_tri(feat, label)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(label).sum().item() / 2)

            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update P
            train_loss.update(loss.item(), 2 * input.size(0))
            id_loss.update(loss_ce.item(), 2 * input.size(0))
            tri_loss.update(loss_tri.item(), 2 * input.size(0))
            total += label.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 30 == 0:
                print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                      f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'lr:{current_lr:.4f} '
                      f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      f'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                      f'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                      f'Accu: {100. * correct / total:.2f}')
            # For all batch, write in tensorBoard
        writer.add_scalar('Train_total_loss', train_loss.avg, epoch)
        writer.add_scalar('Train_id_loss', id_loss.avg, epoch)
        writer.add_scalar('Train_tri_loss', tri_loss.avg, epoch)
        # writer.add_scalar('lr', current_lr, epoch)
        writer.add_scalar('Training accuracy', 100. * correct / total, epoch)

    ######################################### TRAINING
    # start_epoch = 0
    loader_batch = batch_num_identities * num_of_same_id_in_batch

    # define loss function
    criterion_id = nn.CrossEntropyLoss().to(device)
    criterion_tri = BatchHardTripLoss(batch_size=loader_batch, margin= 0.3).to(device)

    best_acc = 0
    for epoch in range(81):

        print('==> Preparing Data Loader...')
        # identity sampler - Give iteratively index from a randomized list of color index and thermal index
        if args.train == "visible":
            sampler_train  = UniModalIdentitySampler(trainset.train_color_label, \
                                train_color_pos, \
                                num_of_same_id_in_batch, batch_num_identities)
            trainset.cIndex = sampler_train.index1  # color index
        elif args.train == "thermal":
            sampler_train = UniModalIdentitySampler(trainset.train_thermal_label, \
                                                    train_thermal_pos, \
                                                    num_of_same_id_in_batch, batch_num_identities)
            trainset.tIndex = sampler_train.index1


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                                sampler=sampler_train, num_workers=workers, drop_last=True)


        print(f'len trainloader : {len(trainloader)}')

        # training
        training(epoch)

        ######################################### VALIDATION
        if epoch > 0 and epoch % 2 == 0:
            valid_loss = AverageMeter()
            valid_id_loss = AverageMeter()
            valid_tri_loss = AverageMeter()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            # Prepare valid loader
            if args.train == "training" :
                sampler_valid = UniModalIdentitySampler(validset.valid_color_label, valid_color_pos, \
                                                    num_of_same_id_in_batch, batch_num_identities)
                validset.cIndex = sampler_valid.index1
            if args.train == "thermal" :
                sampler_valid = UniModalIdentitySampler(validset.valid_thermal_label, valid_thermal_pos, \
                                                    num_of_same_id_in_batch, batch_num_identities)
                validset.tIndex = sampler_valid.index1


            validloader = torch.utils.data.DataLoader(validset, batch_size=loader_batch, \
                                                      sampler=sampler_valid, num_workers=workers, drop_last=True)

            print(f'Validation epoch: {epoch}')
            correct = 0
            total = 0
            # LOADER CORRESPONDING TO VISIBLE OR THERMAL DUE TO args.train
            with torch.no_grad():
                for batch_idx, (input, label) in enumerate(validloader):
                    input = Variable(input.cuda())
                    label = Variable(label.cuda())
                    # input = Variable(input)
                    # label = Variable(label)
                    feat, out0, = net(input)  # Call the visible branch only
                    # feat, out0, = net(input, input, modal=1) # Call the visible branch only

                    loss_ce = criterion_id(out0, label)
                    loss_tri, batch_acc = criterion_tri(feat, label)
                    correct += (batch_acc / 2)
                    _, predicted = out0.max(1)
                    correct += (predicted.eq(label).sum().item() / 2)
                    loss = loss_ce + loss_tri
                    total += label.size(0)
                    acc = 100. * correct / total

                    valid_loss.update(loss.item(), 2 * input.size(0))
                    valid_id_loss.update(loss_ce.item(), 2 * input.size(0))
                    valid_tri_loss.update(loss_tri.item(), 2 * input.size(0))

                    print(f'Loss: {loss:.4f}'
                          f'iLoss: {loss_ce:.4f}  '
                          f'TLoss: {loss_tri:.4f}  '
                          f'Accurac= {acc}'
                          )
            # save model
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                state = {
                    'net': net.state_dict(),
                    'loss': loss,
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            writer.add_scalar('Valid_total_loss', valid_loss.avg, epoch)
            writer.add_scalar('Valid_loss', valid_id_loss.avg, epoch)
            writer.add_scalar('Valid_tri_loss', valid_tri_loss.avg, epoch)
            writer.add_scalar('Validation accuracy', acc, epoch)

if __name__ == '__main__':
    freeze_support()
    multi_process()