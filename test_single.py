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
from evaluation import eval_regdb
from torchvision import transforms
import torch.utils.data
from multiprocessing import freeze_support

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

nclass = 164
nclass = 395
# net = Network(class_num=nclass).to(device)

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--train', default='visible', help='train visible or thermal only')
args = parser.parse_args()

pool_dim = 2048
# Init variables :
img_w = 144
img_h = 288
test_batch_size = 64
batch_num_identities = 16  # 16 different identities in a batch
num_of_same_id_in_batch = 4  # Number of same identity in a batch
workers = 4
lr = 0.001
checkpoint_path = '../save_model/'
data_path = '../Datasets/RegDB/'

if args.dataset == "sysu":
    if args.train == 'visible':
        suffix = f'RegDB_person_Visible_only_sysu({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
    elif args.train == "thermal":
        suffix = f'RegDB_person_Thermal_only_sysu({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
if args.dataset == "regdb":
    if args.train == 'visible':
        suffix = f'RegDB_person_Visible_only_regdb({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
    elif args.train == "thermal":
        suffix = f'RegDB_person_Thermal_only_regdb({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

print(f'Testing {args.train} ReID')
# suffix = f'RegDB_person_Visible({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])

def extract_gall_feat(gall_loader, ngall, net, visible_train = False):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            # input = Variable(input)
            input = Variable(input.cuda())

            feat_pool, feat_fc = net(input)
            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader, nquery, net):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            # input = Variable(input)
            feat_pool, feat_fc = net(input)
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc

def multi_process() :

        end = time.time()

        #model_path = checkpoint_path +  args.resume
        model_path = '../save_model/' + suffix + '_best.t'
        # model_path = checkpoint_path + 'regdb_awg_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        if os.path.isfile(model_path):

            print(f'==> Loading {args.train} checkpoint..')

            checkpoint = torch.load(model_path)
            net = Network(class_num=nclass)
            net.to(device)
            net.load_state_dict(checkpoint['net'])
        else :
            print("Saved model not loaded, care")
            net = Network(class_num = nclass).to(device)
        # Building test set and data loaders

        if args.dataset == "regdb" :
            query_img, query_label, gall_img, gall_label = process_test_regdb(data_path, modal=args.train, split=True)
        elif args.dataset == "sysu" :
            ir_img, ir_id, vis_img, vis_id = process_test_sysu(data_path)
            vis_pos, ir_pos  = GenIdx(vis_id, ir_id)
            if args.train == "visible" :
                query_img, query_label, gall_img, gall_label = \
                process2_test_sysu(data_path, modal=args.train, vis_img=vis_img, vis_id=vis_id, vis_pos=vis_pos )
            if args.train == "thermal" :
                query_img, query_label, gall_img, gall_label = \
                process2_test_sysu(data_path, modal=args.train, ir_img =ir_img , ir_id=ir_id, ir_pos = ir_pos)


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

        # if True = thermal to visible, else, the reverse
        if True :
            # pool5 feature

            distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

            # fc feature
            distmat = np.matmul(gall_feat_fc , np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat,gall_label,  query_label )
        else:
            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)


        print('==> Test results:')
        print(
            f'FC:     Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%}| Rank-20: {cmc[19]:.2%}| mAP: {mAP:.2%}| mINP: {mINP:.2%}')
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))



if __name__ == '__main__':
    freeze_support()
    multi_process()