import numpy as np
import sys
import torch.nn as nn
import torch


# t = torch.rand(4, 4)


# print(t.size(0))
# print(t.size(1))
# b = t.view(t.size(0), t.size(1))
#
# print(t)
# print(b)
# sys.exit()
#
gall_feat_pool = np.zeros((2, 5))
gall_feat_pool[0,0] = 2.
gall_feat_pool[0,1] = 2
gall_feat_pool[0,2] = 3.2
gall_feat_pool[0,3] = 3.2
gall_feat_pool[1,0] = 0.
gall_feat_pool[1,1] = 0
gall_feat_pool[1,2] = 3.2
gall_feat_pool[1,3] = 3.2
a = np.array([[1,3,4],
       [4,1,2 ],
              [2,2,12]])
b = np.array([[0,1,2],
       [1,2,0],
              [0,1,2]])
# a = np.argsort(-a, axis = 1)
print(a[b])
sys.exit()
print(gall_feat_pool[0])
print(gall_feat_pool[1])
print(gall_feat_pool.shape)
dist = np.linalg.norm(gall_feat_pool[0]-gall_feat_pool[1])
print(dist)
sys.exit()
gall_feat_pool = torch.tensor(gall_feat_pool)
criterion_MSE = nn.MSELoss()
print(criterion_MSE(gall_feat_pool[1,:],gall_feat_pool[0,:]))
print(gall_feat_pool)
# print(gall_feat_pool[:,1])
# print(gall_feat_pool.shape[0])
# print(gall_feat_pool)
# print(np.argsort(gall_feat_pool, axis=1))
# print(gall_feat_pool[:,1])
# print( gall_feat_pool[:,2])
# print( (gall_feat_pool[:,1] == gall_feat_pool[:,2]))
# print( (gall_feat_pool[:,1] == gall_feat_pool[:,2]).astype(np.int32))
# print(True.astype(np.int32))
sys.exit()
dist = np.linalg.norm(gall_feat_pool[1, :] - gall_feat_pool[0, :])

print(dist)
print(gall_feat_pool)
print(gall_feat_pool[0, :])
gallery, query = np.array([0])

matrix = np.zeros((gallery.shape[0], query.shape[1]))
for k in range(gallery.shape[0]):
    for i in range(query.shape[1]):
        matrix[k,i] = np.linalg.norm(gallery[k] - query[:,i])



# gall_feat_pool[0, :] = [0.2,0.3, 0.7, 0.9, 0.6]
# gall_feat_pool[1, :] = [4,2,3,1,5]
#
# print(-gall_feat_pool)
#
# print(np.argsort(-gall_feat_pool, axis = 1))

# List = [0,0,0,0,1,2,3]
# nw = [6,6,4,4,4,4,4]
#
#
# List = np.array(List)
# nw = np.array(nw)
# print(List[nw])

