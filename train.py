import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import time
from data_loader import RegDBData, GenIdx, process_test_regdb, TestData
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import IdentitySampler, AverageMeter, adjust_learning_rate
from loss import BatchHardTripLoss
from tensorboardX import SummaryWriter
from model import Network
from multiprocessing import freeze_support
from test import extract_gall_feat, extract_query_feat
from evaluation import eval_regdb
import sys

def multi_process() :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter("runs/CrossModal1")


    # Init variables :
    img_w = 144
    img_h = 288
    test_batch_size = 64
    batch_num_identities = 8 # 8 different identities in a batch
    num_of_same_id_in_batch = 4 # Number of same identity in a batch
    workers = 4
    lr = 0.001
    checkpoint_path = '../save_model/'
    suffix = f'RegDB_person({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
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

    Timer1 = time.time()
    ######################################### TRAIN SET
    trainset = RegDBData(data_path, trial = 1, transform=transform_train)

    # generate the idx of each person identity for instance, identity 10 have the index 100 to 109
    # It is a list of list color_pos[10] = [100, ..., 109]
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    ######################################### TEST SET
    # First import
    query_img, query_label = process_test_regdb(data_path, trial= 1, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial= 1, modal='thermal')
    # Gallery of thermal images - Queryset = Gallery of visible query
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=( img_w, img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=( img_w, img_h))
    # Test data loader
    gall_loader = torch.utils.data.DataLoader(gallset, batch_size= test_batch_size, shuffle=False, num_workers= workers)
    query_loader = torch.utils.data.DataLoader(queryset, batch_size= test_batch_size, shuffle=False, num_workers= workers)

    n_class = len(np.unique(trainset.train_color_label))
    n_query = len(query_label)
    n_gall = len(gall_label)

    print('Dataset RegDB statistics:')
    print('   set     |  Nb ids |  Nb img    ')
    print('  ------------------------------')
    print(f'  visible  | {n_class:5d} | {len(trainset.train_color_label):8d}')
    print(f'  thermal  | {n_class:5d} | {len(trainset.train_thermal_label):8d}')
    print('  ------------------------------')
    print(f'  query    | {len(np.unique(query_label)):5d} | {n_query:8d}')
    print(f'  gallery  | {len(np.unique(gall_label)):5d} | {n_gall:8d}')
    print('  ------------------------------')
    print(f'Data Loading Time:\t {time.time() - Timer1:.3f}')
    print(' ')
    print('==> Building model..')

    ######################################### MODEL

    net = Network(n_class).to(device)

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

    def train(epoch):

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

        for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

            labels = torch.cat((label1, label2), 0)

            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            labels = Variable(labels.cuda())

            data_time.update(time.time() - end)

            #feat is the feature vectore out of
            # Out is the last output
            feat, out0, = net(input1, input2)

            loss_ce = criterion_id(out0, labels)
            loss_tri, batch_acc = criterion_tri(feat, labels)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)

            loss = loss_ce + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update P
            train_loss.update(loss.item(), 2 * input1.size(0))
            id_loss.update(loss_ce.item(), 2 * input1.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))
            total += labels.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 30 == 0:
                print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                      f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'lr:{current_lr:.3f} '
                      f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      f'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                      f'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                      f'Accu: {100. * correct / total:.2f}')
        # For each batch, write in tensorBoard
        writer.add_scalar('total_loss', train_loss.avg, epoch)
        writer.add_scalar('id_loss', id_loss.avg, epoch)
        writer.add_scalar('tri_loss', tri_loss.avg, epoch)
        writer.add_scalar('lr', current_lr, epoch)

    def test(epoch):

        #Get all normalized distance
        gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader, n_gall, net = net)
        query_feat_pool, query_feat_fc = extract_query_feat(query_loader, n_query, net = net)

        start = time.time()
        # compute the similarity
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        # evaluation

        cmc, mAP, mINP = eval_regdb(-distmat_pool, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_fc, query_label, gall_label)

        print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

        writer.add_scalar('rank1', cmc[0], epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('mINP', mINP, epoch)
        writer.add_scalar('rank1_att', cmc_att[0], epoch)
        writer.add_scalar('mAP_att', mAP_att, epoch)
        writer.add_scalar('mINP_att', mINP_att, epoch)
        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att

    # Training part
    # start_epoch = 0
    loader_batch = batch_num_identities * num_of_same_id_in_batch
    # define loss function
    criterion_id = nn.CrossEntropyLoss().to(device)
    criterion_tri = BatchHardTripLoss(batch_size=loader_batch, margin= 0.3).to(device)
    best_acc = 0
    # for epoch in range(start_epoch, 81 - start_epoch):
    for epoch in range(81):

        print('==> Preparing Data Loader...')
        # identity sampler - Give iteratively index from a randomized list of color index and thermal index
        sampler = IdentitySampler(trainset.train_color_label, \
                                  trainset.train_thermal_label, color_pos, thermal_pos, num_of_same_id_in_batch, batch_num_identities)

        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index
        # print(epoch)
        # print(trainset.cIndex)
        # print(trainset.tIndex)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                                sampler=sampler, num_workers=workers, drop_last=True)
        print(len(trainloader))
        # training
        train(epoch)

        if epoch > 0 and epoch % 2 == 0:
            print(f'Test Epoch: {epoch}')

            # testing
            cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
            # save model
            if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
                best_acc = cmc_att[0]
                best_epoch = epoch
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc_att,
                    'mAP': mAP_att,
                    'mINP': mINP_att,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            # save model
            # if epoch > 10 and epoch % args.save_epoch == 0:
            #     state = {
            #         'net': net.state_dict(),
            #         'cmc': cmc,
            #         'mAP': mAP,
            #         'epoch': epoch,
            #     }
            #     torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

            print(
                'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
            print('Best Epoch [{}]'.format(best_epoch))

if __name__ == '__main__':
    freeze_support()
    multi_process()