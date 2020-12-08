import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import argparse
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
from evaluation import eval_regdb
import sys
from test_single import extract_gall_feat, extract_query_feat
from data_augmentation import data_aug

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def multi_process() :
    device = 'cpu'
    # device2 = 'cuda' if torch.cuda.is_available() else 'cpu'


    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
    # parser.add_argument('--trained', default='visible', help='train visible or thermal only')
    parser.add_argument('--board', default='default', help='tensorboard name')
    parser.add_argument('--distilled', default='VtoT', help='tensorboard name')
    parser.add_argument('--mode', default='cp&freeze', help='tensorboard name')

    args = parser.parse_args()

    writer = SummaryWriter(f"runs/{args.board}")
    # Init variables :
    img_w = 144
    img_h = 288
    test_batch_size = 64
    batch_num_identities = 16 # 8 different identities in a batch
    trainV_batch_num_identities = 16 # 16 different identities in a batch
    num_of_same_id_in_batch = 4 # Number of same identity in a batch
    workers = 4
    lr = 0.001
    if args.distilled=="VtoT" :
        args.trained="VtoV"
    elif args.distilled=="TtoV":
        args.trained="TtoT"

    suffix = f'{args.dataset}_person_{args.trained}_only_({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

    checkpoint_path = '../save_model/'

    suffix_distilled = f'{args.dataset}_{args.distilled}_distilled({num_of_same_id_in_batch})_same_id({trainV_batch_num_identities})_lr_{lr}'

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


    # for i in range(6):
    #      plt.subplot(2, 3, i + 1)
    #      plt.imshow(final_train_data[i])
    #      # plt.imshow(final_train_data[i], cmap='gray')
    # plt.show()
    ######################################### TRAIN SET
    Timer1 = time.time()
    print('==> Loading images..')

    #Get Train set and test set
    data_path = '../Datasets/RegDB'

    trainset = RegDBData(data_path, transform=transform_train, modal=args.distilled)

    ######################################### TEST SET
    query_img, query_label, gall_img, gall_label = process_test_regdb(data_path, trial=1, modal=args.distilled)

    # Gallery of thermal images - Queryset = Gallery of visible query
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=( img_w, img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=( img_w, img_h))
    # Test data loader
    gall_loader = torch.utils.data.DataLoader(gallset, batch_size= test_batch_size, shuffle=False, num_workers= workers)
    query_loader = torch.utils.data.DataLoader(queryset, batch_size= test_batch_size, shuffle=False, num_workers= workers)

    n_class = len(np.unique(trainset.train_color_label))
    n_query = len(query_label)
    n_gall = len(gall_label)

    ######################################### VALIDATION SET
    #validset = RegDBData(data_path, transform=transform_train, split="validation", modal="both")
    # print(validset.valid_color_label)
    # loaded_img = len(trainset.train_color_image) + len(validset.valid_color_label) + \
    #         len(trainset.train_thermal_image) + len(validset.valid_thermal_image)

    loaded_img = len(trainset.train_color_image) + len(trainset.train_thermal_image)
    train_color_pos, train_thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    print(f'Data Loading Time:\t {time.time() - Timer1}')
    print(f'Loaded images : {loaded_img}')
    print(f'')
    # Get position list
    # valid_color_pos, valid_thermal_pos = GenIdx(validset.valid_color_label, validset.valid_thermal_label)
    print(' ')
    print('Generated dataset statistics:')
    print('   set     |  Nb ids |  Nb img    ')
    print('  ------------------------------')
    print(f'  Visible  | {len(np.unique(trainset.train_color_label)):5d} | {len(trainset.train_color_label):8d}')
    print(f'  Thermal  | {len(np.unique(trainset.train_thermal_label)):5d} | {len(trainset.train_thermal_label):8d}')
    print('  ------------------------------')
    print(f'  query    | {len(np.unique(query_label)):5d} | {n_query:8d}')
    print(f'  gallery  | {len(np.unique(gall_label)):5d} | {n_gall:8d}')
    print('  ------------------------------')
    print(' ')
    print('==> Building model..')

    ######################################### MODEL

    model_path = '../save_model/' + suffix + '_best.t'

    nclass = len(np.unique(trainset.train_color_label))
    if os.path.isfile(model_path):
        print(f'==> loading checkpoint')
        checkpoint = torch.load(model_path)
        # No weight init for thermal
        net_thermal = Network(class_num=nclass)
        net_thermal.to(device)

        net_visible = Network(class_num=nclass)
        net_visible.to(device)
        if args.mode in ["cp&freeze", "copy"] :
            net_thermal.load_state_dict(checkpoint['net'])
            net_visible.load_state_dict(checkpoint['net'])
        elif args.distilled == "VtoT" :
            net_visible.load_state_dict(checkpoint['net'])
        elif args.distilled == "TtoV" :
            net_thermal.load_state_dict(checkpoint['net'])
    else:
        print("Saved model not loaded, care")
        sys.exit()

    net_visible.train()
    net_thermal.train()
    # Freeze some in thermal model

    if args.mode == "cp&freeze" :

        if args.distilled == "VtoT" :
            net_thermal.Resnet_module.res.layer2.requires_grad = False
            net_thermal.Resnet_module.res.layer3.requires_grad = False
            net_thermal.Resnet_module.res.layer4.requires_grad = False
            print("Several layers frozen")
        elif args.distilled == "TtoV" :
            net_visible.Resnet_module.res.layer2.requires_grad = False
            net_visible.Resnet_module.res.layer3.requires_grad = False
            net_visible.Resnet_module.res.layer4.requires_grad = False
            print("Several layers frozen")


    ######################################### TRAINING
    print('==> Start Training...')

    #Train function
    ignored_params = list(map(id, net_thermal.bottleneck.parameters())) \
                     + list(map(id, net_thermal.fc.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net_thermal.parameters())
    base_params_v = filter(lambda p: id(p) not in ignored_params, net_visible.parameters())

    optimizer_thermal = optim.SGD([
        {'params': base_params, 'lr': 0.1 * lr},
        {'params': net_thermal.bottleneck.parameters(), 'lr': lr},
        {'params': net_thermal.fc.parameters(), 'lr': lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    #Train function
    ignored_params = list(map(id, net_visible.bottleneck.parameters())) \
                     + list(map(id, net_visible.fc.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net_visible.parameters())

    optimizer_visible = optim.SGD([
        {'params': base_params_v, 'lr': 0.1 * lr},
        {'params': net_visible.bottleneck.parameters(), 'lr': lr},
        {'params': net_visible.fc.parameters(), 'lr': lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    ################FUNCTIONs :


    def train_thermal(epoch):
        if args.distilled == "VtoT" :
            current_lr = adjust_learning_rate(optimizer_thermal, epoch, lr=lr)
        if args.distilled == "TtoV" :
            current_lr = adjust_learning_rate(optimizer_visible, epoch, lr=lr)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0

        end = time.time()
        for batch_idx, (visible_input, thermal_input, visible_label, thermal_label) in enumerate(trainloader):
            # visible_input = Variable(visible_input.cuda())
            # thermal_input = Variable(thermal_input.cuda())
            # visible_label = Variable(visible_label.cuda())
            # thermal_label = Variable(thermal_label.cuda())
            #
            # labels = torch.cat((visible_label, thermal_label), 0)
            visible_input = Variable(visible_input)
            thermal_input = Variable(thermal_input)
            visible_label = Variable(visible_label)
            thermal_label = Variable(thermal_label)
            # labels = Variable(labels)
            data_time.update(time.time() - end)

            feat1, out1, = net_visible(visible_input)  # Call the visible trained net
            feat2, out2, = net_thermal(thermal_input)  # Call the  net thermal to train net

            loss_MSE = criterion_MSE(out1, out2)

            if args.distilled == "VtoT" :
                _, predicted = out2.max(1)
                correct += (predicted.eq(thermal_label).sum().item())
                # correct += (predicted.eq(labels).sum().item())

                optimizer_thermal.zero_grad()
                loss_MSE.backward()
                optimizer_thermal.step()

                # update P
                train_loss.update(loss_MSE.item(), 2 * visible_input.size(0))
                total += thermal_label.size(0)
            elif args.distilled == "TtoV" :
                _, predicted = out1.max(1)
                correct += (predicted.eq(visible_label).sum().item())

                optimizer_visible.zero_grad()
                loss_MSE.backward()
                optimizer_visible.step()

                # update P
                train_loss.update(loss_MSE.item(), 2 * visible_input.size(0))
                # total += visible_label.size(0)
                total += visible_label.size(0)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 30 == 0:
                print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                      f'Time: {batch_time.val:.3f}) '
                      f'lr:{current_lr:.4f} '
                      f'Loss: {train_loss.val:.4f}) '
                      f'Accu: {100. * correct / total:.2f}')
            # For all batch, write in tensorBoard
        writer.add_scalar('Training loss (MSE)', train_loss.avg, epoch)
        writer.add_scalar('Training lr', current_lr, epoch)
        writer.add_scalar('Training accuracy ', 100. * correct / total, epoch)


    def test(epoch):

        end = time.time()
        #Get all normalized distance
        if args.distilled== "VtoT" :
            gall_feat_pool, gall_feat_fc = extract_gall_feat(gall_loader, n_gall, net = net_thermal)
            query_feat_pool, query_feat_fc = extract_query_feat(query_loader, n_query, net = net_visible)
        print(f"Feature extraction time : {time.time() - end}")
        start = time.time()
        # compute the similarity
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        # evaluation
        if args.dataset == 'regdb':
            cmc, mAP, mINP      = eval_regdb(-distmat_pool, query_label, gall_label)
            cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_fc, query_label, gall_label)
        #
        # elif args.dataset == 'sysu':
        #
        #     cmc, mAP, mINP = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)
        #     cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_fc, query_label, gall_label, query_cam, gall_cam)

        print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
        writer.add_scalar('Accuracy validation', mAP, epoch)

        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att

    # Training part
    # start_epoch = 0
    loader_batch = batch_num_identities * num_of_same_id_in_batch
    # define loss function
    criterion_MSE = nn.MSELoss().to(device)

    best_acc = 0
    for epoch in range(81):

        print('==> Preparing Data Loader...')
        # Prepare training loader :
        sampler_train = IdentitySampler_paired(trainset.train_color_label, trainset.train_thermal_label, \
                                        train_color_pos, train_thermal_pos, \
                                        num_of_same_id_in_batch, batch_num_identities)
        trainset.cIndex = sampler_train.index1  # color index
        trainset.tIndex = sampler_train.index2 # thermal index

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=loader_batch, \
                                                  sampler=sampler_train, num_workers=workers, drop_last=True)
        print(f'len trainloader : {len(trainloader)}')

        # training
        train_thermal(epoch)
        # validation :
        if epoch > 0 and epoch % 2 == 0  :
            print(f'Test Epoch: {epoch}')

            # testing
            cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
            # save model
            if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
                best_acc = cmc_att[0]
                best_epoch = epoch
                state = {
                    'net': net_thermal.state_dict(),
                    'cmc': cmc_att,
                    'mAP': mAP_att,
                    'mINP': mINP_att,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix_distilled + '_best.t')

            # save model
            if epoch > 10 and epoch % 20 == 0:
                state = {
                    'net': net_thermal.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix_distilled + '_best.t')
            print(
                'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
            print('Best Epoch [{}]'.format(best_epoch))
        # if epoch > 0 and epoch % 2 == 0 and False:
        if False :

            valid_loss = AverageMeter()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            print(f'Validation epoch: {epoch}')
            # Prepare valid loader
            sampler_valid = IdentitySampler_paired(validset.valid_color_label, validset.valid_thermal_label, \
                                            valid_color_pos, valid_thermal_pos, \
                                            num_of_same_id_in_batch, batch_num_identities)

            validset.cIndex = sampler_valid.index1
            validset.tIndex = sampler_valid.index2

            validloader = torch.utils.data.DataLoader(validset, batch_size=loader_batch, \
                                                      sampler=sampler_valid, num_workers=workers, drop_last=True)
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (visible_input, thermal_input, visible_label, thermal_label) in enumerate(validloader):
                    # visible_input = Variable(visible_input.cuda())
                    # thermal_input = Variable(thermal_input.cuda())
                    # visible_label = Variable(visible_label.cuda())
                    # thermal_label = Variable(thermal_label.cuda())
                    labels = torch.cat((visible_label, thermal_label), 0)
                    visible_input = Variable(visible_input)
                    thermal_input = Variable(thermal_input)
                    visible_label = Variable(visible_label)
                    thermal_label = Variable(thermal_label)

                    feat1, out1, = net_visible(visible_input)  # Call the visible branch only
                    feat2, out2 = net_thermal(thermal_input)  # Call the visible branch only

                    loss_MSE = criterion_MSE(out1, out2)
                    _, predicted = out2.max(1)
                    #correct += (predicted.eq(thermal_label).sum().item())
                    correct += (predicted.eq(labels).sum().item())

                    # total += visible_label.size(0)
                    total += labels.size(0)
                    acc = 100. * correct / total

                    valid_loss.update(loss_MSE.item(), 2 * visible_input.size(0))

                    print(f'MSE Loss: {loss_MSE:.4f}  '
                          f'Validation accuracy= {acc}'
                          )

            # save model
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                state = {
                    'net_thermal': net_thermal.state_dict(),
                    'loss': loss_MSE,
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix_distilled + '_best.t')

            writer.add_scalar('Validation loss (MSE)', valid_loss.avg, epoch)
            writer.add_scalar('Validation accuracy', acc, epoch)

if __name__ == '__main__':
    freeze_support()
    multi_process()