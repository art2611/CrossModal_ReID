import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import time
from data_loader import *
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from loss import BatchHardTripLoss
from tensorboardX import SummaryWriter
from model import Network
from multiprocessing import freeze_support
import os
from test import extract_gall_feat, extract_query_feat
from evaluation import eval_regdb
import sys
from data_augmentation import data_aug


def multi_process() :
    device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter("runs/CrossModal3")

    # Init variables :
    img_w = 144
    img_h = 288
    test_batch_size = 64
    batch_num_identities = 8 # 8 different identities in a batch
    trainV_batch_num_identities = 16 # 16 different identities in a batch
    num_of_same_id_in_batch = 4 # Number of same identity in a batch
    workers = 4
    lr = 0.001
    nclass = 206
    checkpoint_path = '../save_model/'
    suffix = f'RegDB_person_Thermal({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
    # Data info  :
    data_path = '../Datasets/RegDB/'
    #log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
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

    # testset = RegDBData_split(data_path, transform=transform_train, split="testing")

    # print(f'Image loaded : {len(np.unique(trainset.train_color_label))}')
    # print(f'len(img valid) {len(np.unique(validset.valid_color_label))}')
    # print(f'len(img train) {len(np.unique(testset.test_color_label))}')

    # Import testset (rpz 20% of the data)
    # testset = RegDBData_split(data_path, transform = transform_test, split="testing")



    # generate the idx of each person identity for instance, identity 10 have the index 100 to 109
    # It is a list of list train_color_pos[10] = [100, ..., 109]
    # train_color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)


    # for i in range(6):
    #      plt.subplot(2, 3, i + 1)
    #      plt.imshow(final_train_data[i])
    #      # plt.imshow(final_train_data[i], cmap='gray')
    # plt.show()
    ######################################### TRAIN SET
    Timer1 = time.time()
    print('==> Loading images..')

    #Get Train set and test set
    trainset = RegDBData_split(data_path, transform=transform_train, split="training")

    ######################################### VALIDATION SET
    validset = RegDBData_split(data_path, transform=transform_train, split="validation")
    # print(validset.valid_color_label)
    valid_Vcolor_pos, validTcolor_pos = GenIdx(validset.valid_color_label, validset.valid_thermal_label)
    loaded_img = len(trainset.train_color_image) + len(validset.valid_color_label) + \
            len(validset.valid_color_image) + len(validset.valid_thermal_image)

    print(f'Loaded images : {loaded_img}')
    print(' ')
    ######################################### Image GENERATION
    print('==> Image generation..')
    # trainset.train_color_image, trainset.train_color_label, _, _ =\
    #     data_aug(visible_images = trainset.train_color_image, Visible_labels = trainset.train_color_label, \
    #     Thermal_images = trainset.train_thermal_image, Thermal_labels = trainset.train_thermal_label)
    loaded_img = len(trainset.train_color_image) + len(validset.valid_color_image) + \
                 len(trainset.train_thermal_image) + len(validset.valid_thermal_image)
    print(f'New image number : {loaded_img}')
    train_color_pos, thermal_color_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    print(f'Identities number : {len(train_color_pos)}')
    print(' ')
    print('New dataset statistics:')
    print('   set     |  Nb ids |  Nb img    ')
    print('  ------------------------------')
    print(f'  train_Visible  | {len(np.unique(trainset.train_color_label)):5d} | {len(trainset.train_color_label):8d}')
    print(f'  train_Thermal  | {len(np.unique(trainset.train_thermal_label)):5d} | {len(trainset.train_thermal_label):8d}')
    print(f'  valid_Visible  | {len(np.unique(validset.valid_color_label)):5d} | {len(validset.valid_color_label):8d}')
    print(f'  valid_Thermal  | {len(np.unique(validset.valid_thermal_label)):5d} | {len(validset.valid_thermal_label):8d}')
    print('  ------------------------------')
    print(f'Data Loading Time:\t {time.time() - Timer1:.3f}')
    print(' ')
    print('==> Building model..')

    ######################################### MODEL
    suffix_visible = f'RegDB_person_Visible({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}_best.t'
    model_path = checkpoint_path + suffix_visible
    if os.path.isfile(model_path):
        print('==> loading checkpoint')
        checkpoint = torch.load(model_path)
        net_thermal = Network(class_num=nclass)
        net_thermal.to(device)
        net_thermal.load_state_dict(checkpoint['net'])
        net_visible = Network(class_num=nclass)
        net_visible.to(device)
        net_visible.load_state_dict(checkpoint['net'])
    else:
        print("Saved model not loaded, care")
        sys.exit()

    # Freeze visible model
    net_visible.eval()
    # Freeze some in thermal model
    for param in net_thermal.parameters():
        # param.requires_grad = False
        print(param)
    sys.exit()

    ######################################### TRAINING
    print('==> Start Training...')

    #Train function
    ignored_params = list(map(id, net_thermal.bottleneck.parameters())) \
                     + list(map(id, net_thermal.fc.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net_thermal.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr},
        {'params': net_thermal.bottleneck.parameters(), 'lr': lr},
        {'params': net_thermal.fc.parameters(), 'lr': lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    ################FUNCTIONs :


    def train_visible(epoch):
        current_lr = adjust_learning_rate(optimizer, epoch, lr=lr)
        train_loss = AverageMeter()
        id_loss = AverageMeter()
        tri_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0

        # switch to train mode
        net_thermal.train()
        end = time.time()
        for batch_idx, (visible_input, visible_label) in enumerate(trainloader):
            # visible_input = Variable(visible_input.cuda())
            # visible_label = Variable(visible_label.cuda())
            visible_input = Variable(visible_input)
            visible_label = Variable(visible_label)

            data_time.update(time.time() - end)

            # feat is the feature vector out of
            # Out is the last output
            feat, out0, = net_thermal(visible_input, visible_input, modal=1)  # Call the visible branch only

            loss_ce = criterion_id(out0, visible_label)
            loss_tri, batch_acc = criterion_tri(feat, visible_label)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(visible_label).sum().item() / 2)

            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update P
            train_loss.update(loss.item(), 2 * visible_input.size(0))
            id_loss.update(loss_ce.item(), 2 * visible_input.size(0))
            tri_loss.update(loss_tri.item(), 2 * visible_input.size(0))
            total += visible_label.size(0)

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
        writer.add_scalar('total_loss', train_loss.avg, epoch)
        writer.add_scalar('id_loss', id_loss.avg, epoch)
        writer.add_scalar('tri_loss', tri_loss.avg, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        writer.add_scalar('acc_train', 100. * correct / total, epoch)


    # Training part
    # start_epoch = 0
    loader_batch = batch_num_identities * num_of_same_id_in_batch
    # define loss function
    criterion_id = nn.CrossEntropyLoss().to(device)
    criterion_tri = BatchHardTripLoss(batch_size=loader_batch, margin= 0.3).to(device)


    #Prepare valid loader
    sampler_valid = UniModalIdentitySampler(validset.valid_color_label, valid_color_pos, \
                                           num_of_same_id_in_batch, trainV_batch_num_identities)

    # print(f'trainset.train_color_label {len(validset.valid_color_label)}')
    # print(f'trainset.train_color_label {len(np.unique(validset.valid_color_label))}')
    # print(f'len train_color_pos {len(valid_color_pos)}')

    validset.cIndex = sampler_valid.index1
    validloader = torch.utils.data.DataLoader(validset, batch_size=loader_batch, \
                            sampler=sampler_valid, num_workers=workers, drop_last=True)

    # print(f'trainset.train_color_label {len(trainset.train_color_label)}')
    # print(f'trainset.train_color_label {len(np.unique(trainset.train_color_label))}')
    # print(f'len train_color_pos {len(train_color_pos)}')


    sampler_train = UniModalIdentitySampler(trainset.train_color_label, train_color_pos, \
                                            num_of_same_id_in_batch, trainV_batch_num_identities)

    trainset.cIndex = sampler_train.index1  # color index
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                                              sampler=sampler_train, num_workers=workers, drop_last=True)


    best_acc = 0
    for epoch in range(81):

        print('==> Preparing Data Loader...')
        # identity sampler - Give iteratively index from a randomized list of color index and thermal index
        sampler_train  = UniModalIdentitySampler(trainset.train_color_label, \
                                train_color_pos, \
                                num_of_same_id_in_batch, trainV_batch_num_identities)

        trainset.cIndex = sampler_train.index1  # color index
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                                sampler=sampler_train, num_workers=workers, drop_last=True)


        print(f'len trainloader : {len(trainloader)}')

        # training
        train_visible(epoch)
        # validation
        if epoch > 0 and epoch % 1 == 0:
            valid_loss = AverageMeter()
            valid_id_loss = AverageMeter()
            valid_tri_loss = AverageMeter()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            print(f'Validation epoch: {epoch}')
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (visible_input, visible_label) in enumerate(validloader):
                    # visible_input = Variable(visible_input.cuda())
                    # visible_label = Variable(visible_label.cuda())
                    visible_input = Variable(visible_input)
                    visible_label = Variable(visible_label)
                    feat, out0, = net_thermal(visible_input, visible_input, modal=1)  # Call the visible branch only

                    loss_ce = criterion_id(out0, visible_label)
                    loss_tri, batch_acc = criterion_tri(feat, visible_label)
                    correct += (batch_acc / 2)
                    _, predicted = out0.max(1)
                    correct += (predicted.eq(visible_label).sum().item() / 2)
                    loss = loss_ce + loss_tri
                    total += visible_label.size(0)
                    acc = 100. * correct / total

                    valid_loss.update(loss.item(), 2 * visible_input.size(0))
                    valid_id_loss.update(loss_ce.item(), 2 * visible_input.size(0))
                    valid_tri_loss.update(loss_tri.item(), 2 * visible_input.size(0))

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
                    'net_thermal': net_thermal.state_dict(),
                    'loss': loss,
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            writer.add_scalar('Valid_total_loss', valid_loss.avg, epoch)
            writer.add_scalar('Valid_loss', valid_id_loss.avg, epoch)
            writer.add_scalar('Valid_tri_loss', valid_tri_loss.avg, epoch)
            writer.add_scalar('acc_test', acc, epoch)

if __name__ == '__main__':
    freeze_support()
    multi_process()