import numpy as np
from torch.utils.data.sampler import Sampler

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                #On choisit un nombre num pos au hasard de même personne d'identité batchidx[i]
                #On soustraint le nombre de label car l'identité est décalée quand on a par exemple20% de la bdd
                sample_color = np.random.choice(color_pos[batch_idx[i]-train_color_label[0]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]- train_thermal_label[0]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class IdentitySampler_paired(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        #On fait autant de batch que possible
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)

            #Pour chaque batch
            for i in range(batchSize):
                #On choisit un nombre num pos au hasard de même personne d'identité batchidx[i]
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                # sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                sample_thermal =  np.random.choice(color_pos[batch_idx[i]], num_pos)
                # We trial with strict pairs or not, seems a bit better when not strict
                #sample_thermal = sample_color
                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class UniModalIdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """
    def __init__(self, train_label, _pos, num_pos, batchSize):
        uni_label = np.unique(train_label)
        self.n_classes = len(uni_label)
        # print(len(color_pos))
        N = len(train_label)
        for j in range(int(N / (batchSize * num_pos) + 1)):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(_pos[batch_idx[i]], num_pos)
                if j == 0 and i == 0:
                    index1 = sample_color
                else:
                    index1 = np.hstack((index1, sample_color))
        # print(len(index1))
        # print(N)
        self.index1 = index1
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

def RandomIdGenerator(train_color_label, color_pos, num_pos, batchSize):
    #Color pos = Liste de liste d'indices correspondant aux différentes identitées ( color_pos[0] = [0,...,29])
    #num_pos = nbr de personnes de même ids qu'on veut choisir par batch
    #Batchsize = nbr de différentes ids qu'on veut dans un batch
    uni_label = np.unique(train_color_label)
    N = len(train_color_label)
    #70% kept for training
    N_train = int(len(train_color_label) * 70 / 100)
    for j in range(len(train_color_label)):
        batch_idx = np.random.choice(uni_label, batchSize, replace=False)
        for i in range(batchSize):
            sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
            if j == 0 and i == 0:
                index = sample_color
            else:
                index = np.hstack((index, sample_color))
    # print(f"N : {N}")
    # print(f'70% : {N_train}')
    # print(f"len(index) : {len(index)}")
    index_train = []
    index_valid = []
    for k in range(len(index)):
        if k < N_train*64 :
            index_train.append(index[k])
        else :
            index_valid.append(index[k])
    N_valid = len(index_valid)/64
    # print(f'index train nbr : {len(index_train)}')
    # print(f'index test nbr : {len(index_valid)}')
    return(index_train, N_train , index_valid, N_valid)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = lr
    elif epoch >= 20 and epoch < 50:
        lr = lr * 0.1
    elif epoch >= 50:
        lr = lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr
