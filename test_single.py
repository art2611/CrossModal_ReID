import os
import sys
import time
import torch
import argparse
import torch.utils.data
from torch.autograd import Variable
from data_loader import *
import numpy as np
from model import Network
from evaluation import eval_regdb, eval_sysu
from torchvision import transforms
import torch.utils.data
from multiprocessing import freeze_support
from tensorboardX import SummaryWriter
from datetime import date
from extract_feat import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--trained', default='VtoV', help='train visible or thermal only')
parser.add_argument('--reid', default='VtoV', help='test this type of reid with selected trained model')
args = parser.parse_args()


batch_num_identities = 16  # 16 different identities in a batch
num_of_same_id_in_batch = 4  # Number of same identity in a batch
lr = 0.001
suffix = f'{args.dataset}_person_{args.train}_only_({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

#If reid TtoT sur modèle de distillation entraîné
suffix_thermal = f'regdb_VtoT_distilled({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

### Tensorboard init
today = date.today()
d1 = today.strftime("%d")
if args.trained == "VtoV":
    args.trained = "Visible"
elif args.trained == "TtoT" :
    args.trained = "thermal"
writer = SummaryWriter(f"runs/{args.trained}_model_singleReID_{args.reid}-test_{args.dataset}_day{d1}_{time.time()}")

pool_dim = 2048
# Init variables :
img_w = 144
img_h = 288
test_batch_size = 64
workers = 4

checkpoint_path = '../save_model/'

if args.dataset == "sysu":
    nclass = 296
    data_path = '../Datasets/SYSU/'

if args.dataset == "regdb":
    data_path = '../Datasets/RegDB/'
    nclass = 206


print(f'Testing {args.trained} ReID')
# suffix = f'RegDB_person_Visible({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])



def multi_process() :

    end = time.time()

    #model_path = checkpoint_path +  args.resume
    model_path = '../save_model/' + suffix + '_best.t'
    print(model_path)
    # model_path = checkpoint_path + 'regdb_awg_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
    if os.path.isfile(model_path):

        print(f'==> Loading {args.trained} checkpoint on {args.dataset} dataset..')

        checkpoint = torch.load(model_path)
        net = Network(class_num=nclass)
        net.to(device)
        net.load_state_dict(checkpoint['net'])
        print("Model_loaded")
    else :
        sys.exit("Saved model not found")
    # Building test set and data loaders
    if args.dataset == "regdb" :
        for trial in range(1, 11):

            query_img, query_label, gall_img, gall_label = process_test_regdb(data_path, modal=args.reid, trial = trial)

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
            gall_loader = torch.utils.data.DataLoader(gallset, batch_size=test_batch_size, shuffle=False, num_workers=workers)

            nquery = len(query_label)
            ngall = len(gall_label)

            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
            query_loader = torch.utils.data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False, num_workers=4)

            print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

            ############################## DISPLAY QUERY AND GALLERY
            # for i in range(6):
            #      plt.subplot(2, 3, i + 1)
            #      if i < 3 :
            #         plt.title("Gallery")
            #         plt.imshow(np.array(gallset.test_image[i*10]))
            #      else :
            #         plt.title("Query")
            #         plt.imshow(np.array(queryset.test_image[i*10]))
            #      # plt.imshow(final_train_data[i], cmap='gray')
            # plt.show()

            query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery = nquery, net = net)
            gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader, ngall = ngall, net = net)

            # pool5 feature
            distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))

            if args.dataset=="regdb":
                cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)


            # fc feature
            distmat = np.matmul(gall_feat_fc , np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat,gall_label,  query_label )


            if trial == 1 :
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
                all_cmc_pool = cmc_pool
                all_mAP_pool = mAP_pool
                all_mINP_pool = mINP_pool
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP
                all_cmc_pool = all_cmc_pool + cmc_pool
                all_mAP_pool = all_mAP_pool + mAP_pool
                all_mINP_pool = all_mINP_pool + mINP_pool

            print('Test Trial: {}'.format(trial))
            print(
                'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print(
                'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

    if args.dataset == 'sysu':

        # testing set
        if args.reid == "VtoT" or args.reid== "TtoV":
            query_img, query_label, query_cam = process_query_sysu(data_path, "test", mode="all", trial=0, reid=args.reid)
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, "test", mode="all", trial=0, reid=args.reid)

            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
            query_loader = data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False, num_workers=4)
            nquery = len(query_label)
            query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery=nquery, net=net)
        elif args.reid =="VtoV" or args.reid == "TtoT" :
            query_img, query_label, query_cam, gall_img, gall_label, gall_cam =\
                process_test_single_sysu(data_path, "test", trial=0, mode='all', relabel=False, reid=args.reid)
            nquery = len(query_label)

        ngall = len(gall_label)
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
        print("  ------------------------------")
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))



        for trial in range(10):
            # testing set
            if args.reid == "VtoT" or args.reid == "TtoV":
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, "test", mode="all",  trial=trial, reid=args.reid)

                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=test_batch_size, shuffle=False, num_workers=4)
                gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader,ngall = ngall, net = net)
            elif args.reid == "VtoV" or args.reid =="TtoT":
                query_img, query_label, query_cam, gall_img, gall_label, gall_cam = \
                    process_test_single_sysu(data_path, "test", trial=trial, mode='all', relabel=False, reid=args.reid)

                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
                query_loader = data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False, num_workers=4)
                query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery=nquery, net=net)

                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=test_batch_size, shuffle=False, num_workers=4)
                gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader,ngall = ngall, net = net)

            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

            if trial == 0:
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
                all_cmc_pool = cmc_pool
                all_mAP_pool = mAP_pool
                all_mINP_pool = mINP_pool
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP
                all_cmc_pool = all_cmc_pool + cmc_pool
                all_mAP_pool = all_mAP_pool + mAP_pool
                all_mINP_pool = all_mINP_pool + mINP_pool

            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print(
                'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10

    cmc_pool = all_cmc_pool / 10
    mAP_pool = all_mAP_pool / 10
    mINP_pool = all_mINP_pool / 10
    print('All Average:')
    print(
        'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print(
        'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
    for k in range(len(cmc)):
        writer.add_scalar('cmc curve', cmc[k]*100, k + 1)
if __name__ == '__main__':
    freeze_support()
    multi_process()