import os
import torch
import torch.utils.data
from torch.autograd import Variable
import time
from data_loader import *
import numpy as np
from model import Network
from evaluation import eval_regdb
from torchvision import transforms
import torch.utils.data
from multiprocessing import freeze_support
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
nclass = 164
# net = Network(class_num=nclass).to(device)

pool_dim = 2048
# Init variables :
img_w = 144
img_h = 288
test_batch_size = 64
batch_num_identities = 16  # 8 different identities in a batch
num_of_same_id_in_batch = 4  # Number of same identity in a batch
workers = 4
lr = 0.001
checkpoint_path = '../save_model/'
data_path = '../Datasets/RegDB/'
suffix_visible = f'RegDB_person_Visible({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
suffix_thermal = f'RegDB_person_Thermal({num_of_same_id_in_batch})_same_id({batch_num_identities})_lr_{lr}'
#
test_mode = [2, 1]  # visible to thermal

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])

def extract_gall_feat(gall_loader, ngall, net):
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

def multi_process():
    ################ Testing preparation
    #model_path = checkpoint_path +  args.resume
    model_visible_path = '../save_model/' + suffix_visible + '_best.t'
    model_thermal_path = '../save_model/' + suffix_thermal + '_best.t'
    end = time.time()
    if os.path.isfile(model_visible_path):
        print('==> loading checkpoint visible ')
        checkpoint_visible = torch.load(model_visible_path)
        net_visible = Network(class_num=nclass)
        net_visible.to(device)
        net_visible.load_state_dict(checkpoint_visible['net'])
    else :
        print("Problem : Saved visible model not loaded, care")
        sys.exit()
    if os.path.isfile(model_thermal_path):
        print('==> loading checkpoint thermal ')
        checkpoint_thermal = torch.load(model_thermal_path)
        net_thermal = Network(class_num=nclass)
        net_thermal.to(device)
        net_thermal.load_state_dict(checkpoint_thermal['net_thermal'])
    else :
        print("Problem : Saved thermal model not loaded, care")
        sys.exit()

    print(f'Networks Loading Time:\t {time.time() - end:.3f}')

    # Building test set and data loaders
    end = time.time()

    query_img, query_label, gall_img, gall_label = process_test_regdb(data_path, modal='VtoT', split=True)

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(img_w, img_h))
    gall_loader = torch.utils.data.DataLoader(gallset, batch_size=test_batch_size, shuffle=False, num_workers=workers)

    nquery = len(query_label)
    ngall = len(gall_label)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(img_w, img_h))
    query_loader = torch.utils.data.DataLoader(queryset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    ############################## DISPLAY QUERY AND GALLERY
    # for i in range(6):
    #      plt.subplot(2, 3, i + 1)
    #      if i < 3 :
    #         plt.title("Gallery")
    #         plt.imshow(np.array(gallset.test_image[i]))
    #      else :
    #         plt.title("Query")
    #         plt.imshow(np.array(queryset.test_image[i]))
    #      # plt.imshow(final_train_data[i], cmap='gray')
    # plt.show()

    print(f'Data Loading Time:\t {time.time() - end:.3f}')
    print(" ")
    print('==> Feature extraction for queries and gallery')
    end = time.time()
    query_feat_pool, query_feat_fc = extract_query_feat(query_loader, nquery = nquery, net = net_visible)
    gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader, ngall = ngall, net = net_thermal)
    print(f'Feature Extraction Time {time.time() - end}')
    # if True = thermal to visible, else, the reverse
    print(" ")
    print('==> Evaluation : ')

    if True :
        # pool5 feature
        distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
        # print(-distmat_pool[0])
        # print(np.argsort(-distmat_pool, axis = 1)[0])
        print(gall_feat_pool.shape[0])  # Number of gallery images
        print(query_feat_pool.shape[0])  # Number of query images
        distance = np.zeros((gall_feat_pool.shape[0], query_feat_pool.shape[0]))

        for i in range(gall_feat_pool.shape[0]):
            for j in range(gall_feat_pool.shape[0]):
                distance[i][j] = np.linalg(gall_feat_pool[i], query_feat_pool[j])
        print(f'ancient distance : {-distmat_pool[0]}')
        print(f'New distance : {-distance[0]}')
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