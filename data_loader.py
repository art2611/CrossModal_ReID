import numpy as np
from PIL import Image
import torch.utils.data as data
import os
import random
import sys
import math
from torchvision import transforms
import matplotlib.pyplot as plt

class RegDBData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, modal = "both"):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_1.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_1.txt'
        #Load color and thermal images + labels
        color_img_file, color_target = load_data(train_color_list)
        thermal_img_file, thermal_target= load_data(train_thermal_list)
        color_image = []
        color_lab = []
        thermal_image = []
        thermal_lab = []
        #Get real and thermal images with good shape in a list

        for i in range(len(color_img_file)):
            #Visible
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            color_image.append(pix_array)
            color_lab.append(color_target[i])
            #Thermal
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            thermal_image.append(pix_array)
            thermal_lab.append(thermal_target[i])

        color_image = np.array(color_image)
        thermal_image = np.array(thermal_image)
        # Init color images / labels
        self.train_color_image = color_image
        self.train_color_label = color_lab

        # Init themal images / labels
        self.train_thermal_image = thermal_image
        self.train_thermal_label = thermal_lab

        self.transform = transform
        # Prepare index
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.modal = modal

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        if self.modal == "both" or self.modal == "visible" :
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        if self.modal == "both" or self.modal == "thermal":
            img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        if self.modal == "both" :
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2, target1, target2
        elif self.modal == "visible" :
            img1 = self.transform(img1)
            return img1, target1
        elif self.modal == "thermal" :
            img2 = self.transform(img2)
            return img2, target2

    def __len__(self):
        if self.modal == "thermal" :
            return len(self.train_thermal_label)
        return len(self.train_color_label)

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex = None, modal ="visible"):
        data_dir = '../Datasets/SYSU/'
        # Load training images (path) and labels
        # 395 ids sont loadées sur les 491
        color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # Labels
        self.train_color_label = color_label
        self.train_thermal_label = thermal_label
        # BGR to RGB
        self.train_color_image = color_image
        self.train_thermal_image = thermal_image

        self.transform = transform

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.modal = modal

    def __getitem__(self, index):
        # Dataset[i] return images from both modal and the corresponding label
        if self.modal == "both" or self.modal == "visible":
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        elif self.modal == "both" or self.modal == "thermal":
            img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        if self.modal == "both":
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2, target1, target2
        elif self.modal == "visible":
            img1 = self.transform(img1)
            return img1, target1
        elif self.modal == "thermal":
            img2 = self.transform(img2)
            return img2, target2

    def __len__(self):
        if self.modal == "thermal":
            return len(self.train_thermal_label)
        return len(self.train_color_label)

#The two next class were used with custom train and data split
class RegDBData_split(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, modal = "both", split="training" ):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_1.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_1.txt'
        #Load color and thermal images + labels
        color_img_file, color_target = load_data(train_color_list)
        thermal_img_file, thermal_target= load_data(train_thermal_list)
        color_image = []
        color_lab = []
        thermal_image = []
        thermal_lab = []
        #Get real and thermal images with good shape in a list
        if split == "training":
            for i in range(len(color_img_file)):
                if i % 10 < 8 :
                    #Visible
                    img = Image.open(data_dir + color_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    color_image.append(pix_array)
                    color_lab.append(color_target[i])
                    #Thermal
                    img = Image.open(data_dir + thermal_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    thermal_image.append(pix_array)
                    thermal_lab.append(thermal_target[i])

            color_image = np.array(color_image)
            thermal_image = np.array(thermal_image)
            # Init color images / labels
            self.train_color_image = color_image
            self.train_color_label = color_lab

            # Init themal images / labels
            self.train_thermal_image = thermal_image
            self.train_thermal_label = thermal_lab
        if split == "validation" :
            for i in range(len(color_img_file)):
                if i % 10 >= 8:
                    #Visible
                    img = Image.open(data_dir + color_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    color_image.append(pix_array)
                    color_lab.append(color_target[i])
                    #Thermal
                    img = Image.open(data_dir + thermal_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    thermal_image.append(pix_array)
                    thermal_lab.append(thermal_target[i])
            color_image = np.array(color_image)
            thermal_image = np.array(thermal_image)
            # Init color images / labels
            self.valid_color_image = color_image
            self.valid_color_label = color_lab

            # Init themal images / labels
            self.valid_thermal_image = thermal_image
            self.valid_thermal_label = thermal_lab

        self.transform = transform
        # Prepare index
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.modal = modal

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        if hasattr(self, "train_color_image"):
            if self.modal == "both" or self.modal == "visible" :
                img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
            if self.modal == "both" or self.modal == "thermal":
                img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        elif hasattr(self, "valid_color_image") :
            if self.modal == "both" or self.modal == "visible" :
                img1, target1 = self.valid_color_image[self.cIndex[index]], self.valid_color_label[self.cIndex[index]]
            if self.modal == "both" or self.modal == "thermal":
                img2, target2 = self.valid_thermal_image[self.tIndex[index]], self.valid_thermal_label[self.tIndex[index]]

        if self.modal == "both" :
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2, target1, target2
        elif self.modal == "visible" :
            img1 = self.transform(img1)
            return img1, target1
        elif self.modal == "thermal" :
            img2 = self.transform(img2)
            return img2, target2

    def __len__(self):
        if self.modal == "thermal" :
            return len(self.train_thermal_label)
        return len(self.train_color_label)

class SYSUData_split(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex = None, split="training", modal ="visible"):
        data_dir = '../Datasets/SYSU/'
        # Load training images (path) and labels
        # 395 ids sont loadées sur les 491
        color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        color_pos, thermal_pos = GenIdx(color_label, thermal_label)
        _color_image = []
        _color_lab = []
        _thermal_image = []
        _thermal_lab = []
        heightyPercent = 0.8
        if split == "training":
            # Dans chaque liste d'index d'une identité, on prends les 80% premieres images de chaque ids pour train.
            for i in range(len(color_pos)):
                u = len(color_pos[i])
                for j in range(u):
                    if j <= int(u * heightyPercent):
                        _color_image.append(color_image[color_pos[i][j]])
                        _color_lab.append(i)
            for i in range(len(thermal_pos)):
                u = len(thermal_pos[i])
                for j in range(u):
                    if j <= int(u * heightyPercent):
                        _thermal_image.append(thermal_image[thermal_pos[i][j]])
                        _thermal_lab.append(i)
            # Labels
            self.train_color_label = _color_lab
            self.train_thermal_label = _thermal_lab
            # BGR to RGB
            self.train_color_image = _color_image
            self.train_thermal_image = _thermal_image
        if split == "validation":
            # Dans chaque liste d'index d'une identité, on prends les 20% dernières images pour validation.
            for i in range(len(color_pos)):
                u = len(color_pos[i])
                for j in range(u):
                    if j > int(u * heightyPercent):
                        _color_image.append(color_image[color_pos[i][j]])
                        _color_lab.append(i)
            for i in range(len(thermal_pos)):
                u = len(thermal_pos[i])
                for j in range(u):
                    if j > int(u * heightyPercent):
                        _thermal_image.append(thermal_image[thermal_pos[i][j]])
                        _thermal_lab.append(i)
            # Labels
            self.valid_color_label = _color_lab
            self.valid_thermal_label = _thermal_lab
            # BGR to RGB
            self.valid_color_image = _color_image
            self.valid_thermal_image = _thermal_image

        self.transform = transform

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.modal = modal

    def __getitem__(self, index):
        # Dataset[i] return images from both modal and the corresponding label
        if hasattr(self, "train_color_image"):
            if self.modal == "both" or self.modal == "visible":
                img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
            elif self.modal == "both" or self.modal == "thermal":
                img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[
                    self.tIndex[index]]
        elif hasattr(self, "valid_color_image"):
            if self.modal == "both" or self.modal == "visible":
                img1, target1 = self.valid_color_image[self.cIndex[index]], self.valid_color_label[self.cIndex[index]]
            elif self.modal == "both" or self.modal == "thermal":
                img2, target2 = self.valid_thermal_image[self.tIndex[index]], self.valid_thermal_label[
                    self.tIndex[index]]

        if self.modal == "both":
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2, target1, target2
        elif self.modal == "visible":
            img1 = self.transform(img1)
            return img1, target1
        elif self.modal == "thermal":
            img2 = self.transform(img2)
            return img2, target2

    def __len__(self):
        if self.modal == "thermal":
            return len(self.train_thermal_label)
        return len(self.train_color_label)


class RegDBThermalData(data.Dataset):
    def __init__(self, data_dir, transform=None, thermalIndex=None, split="training" ):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_thermal_list = data_dir + 'idx/train_thermal_1.txt'
        #Load color and thermal images + labels
        thermal_img_file, thermal_target = load_data(train_thermal_list)
        thermal_image = []
        thermal_lab = []

        #Get real and thermal images with good shape in a list
        # Training => return 50%
        first50percent = int(len(thermal_img_file) * 50 / 100)
        first50percent -= int(len(thermal_img_file) * 50 / 100) % 10
        first80percent = int(len(thermal_img_file) * 80 / 100)
        first80percent -= int(len(thermal_img_file) * 80 / 100)%10
        if split == "training" :
            for i in range(first80percent):
                if i%10 < 7 :
                    img = Image.open(data_dir + thermal_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    thermal_image.append(pix_array)
                    thermal_lab.append(thermal_target[i])

            thermal_image = np.array(thermal_image)
            # Init color images / labels
            self.train_thermal_image = thermal_image
            self.train_thermal_label = thermal_lab

        if split == "validation" :
            for i in range(first80percent):
                if i%10 >= 7 :
                    img = Image.open(data_dir + thermal_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    thermal_image.append(pix_array)
                    thermal_lab.append(thermal_target[i])

            thermal_image = np.array(thermal_image)

            # Init color images / labels
            self.valid_thermal_image = thermal_image
            self.valid_thermal_label = thermal_lab

        self.transform = transform
        # Prepare index
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        if hasattr(self, "train_thermal_image"):
            img1, target1 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        elif hasattr(self, "valid_thermal_image") :
            img1, target1 = self.valid_thermal_image[self.tIndex[index]], self.valid_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)

        return img1, target1
    def __len__(self):
        if hasattr(self, "train_thermal_image"):
            return len(self.train_thermal_label)
        elif hasattr(self, "valid_thermal_image"):
            return len(self.valid_thermal_label)

def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label

# generate the idx of each person identity for instance, identity 10 have the index 100 to 109
def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos

def process_test_regdb(img_dir, modal='VtoT', trial = 1):

    input_visible_data_path = img_dir + f'idx/test_visible_{trial}.txt'
    input_thermal_data_path = img_dir + f'idx/test_thermal_{trial}.txt'
    if modal == "VtoV" or modal == "VtoT" :
        with open(input_visible_data_path) as f:
            data_file_list = open(input_visible_data_path, 'rt').read().splitlines()
            # Get full list of image and labels
            file_image_visible = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
            file_label_visible = [int(s.split(' ')[1]) for s in data_file_list]
    if modal == "TtoT" or modal == "VtoT":
        with open(input_thermal_data_path) as f:
            data_file_list = open(input_thermal_data_path, 'rt').read().splitlines()
            # Get full list of image and labels
            file_image_thermal = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
            file_label_thermal = [int(s.split(' ')[1]) for s in data_file_list]
    #If required, return half of the dataset in two slice
    if modal == "VtoV" :
        file_image = file_image_visible
        file_label = file_label_visible
    if modal == "TtoT" :
        file_image = file_image_thermal
        file_label = file_label_thermal
    if modal == "TtoT" or modal == "VtoV" :
        first_image_slice_query = []
        first_label_slice_query = []
        sec_image_slice_gallery = []
        sec_label_slice_gallery = []
        #On regarde pour chaque id
        for k in range(len(np.unique(file_label))):
            appeared=[]
            # On choisit cinq personnes en query aléatoirement, le reste est placé dans la gallery (5 images)
            for i in range(5):
                rand = random.choice(file_image[k*10:k*10+9])
                while rand in appeared:
                    rand = random.choice(file_image[k*10:k*10+9])
                appeared.append(rand)
                first_image_slice_query.append(rand)
                first_label_slice_query.append(file_label[k*10])
            #On regarde la liste d'images de l'id k, on récupère les images n'étant pas dans query (5 images)
            for i in file_image[k*10:k*10+10] :
                if i not in appeared :
                    sec_image_slice_gallery.append(i)
                    sec_label_slice_gallery.append(file_label[k*10])
        return(first_image_slice_query, np.array(first_label_slice_query), sec_image_slice_gallery, np.array(sec_label_slice_gallery))
    #Ancienne version, on verra comment on fait ici
    if modal == "VtoT" :
        return (file_image_visible, np.array(file_label_visible), file_image_thermal,
                np.array(file_label_thermal))
    elif modal == "TtoV" :
        return(file_image_thermal, np.array(file_label_thermal), file_image_visible, np.array(file_label_visible))


def process_test_sysu(data_path, mode='all'):
    if mode == 'all':
        ir_cameras = ['cam3', 'cam6']
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        ir_cameras = ['cam3', 'cam6']
        rgb_cameras = ['cam1','cam2']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)
    ir_img = []
    ir_id = []
    ir_cam = []
    vis_img = []
    vis_id = []
    vis_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        ir_img.append(img_path)
        ir_id.append(pid)
        ir_cam.append(camid)

    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        vis_img.append(img_path)
        vis_id.append(pid)
        vis_cam.append(camid)
    return ir_img, np.array(ir_id), vis_img, np.array(vis_id)

def process2_test_sysu(data_path, ir_img=[], ir_id=[], ir_pos=[],  vis_img=[], vis_id=[],vis_pos=[], modal="visible"):
    first_image_slice = []
    first_label_slice = []
    sec_image_slice = []
    sec_label_slice = []
    modality = ['visible', 'thermal', 'VtoT', 'TtoV']
    if modal not in modality :
        sys.exit(f"Error, args.train not in {modality}")
    if modal == "visible":
        file_image = vis_img
        file_label = vis_id
        file_pos = vis_pos
    if modal == "thermal":
        file_image = ir_img
        file_label = ir_id
        file_pos = ir_pos
    if modal == "thermal" or modal == "visible":
        for k in range(len(file_pos)):
            appeared = []
            #On s'assure qu'on a bien 10 images de l'identité k
            if len(file_pos[k]) >=10 :
                #On prend 2 images en query, les 8 autres en gallery
                for i in range(2) :
                    rand = random.choice(file_pos[k])
                    while rand in appeared :
                        rand = random.choice(file_pos[k])
                    appeared.append(rand)
                    first_image_slice.append(file_image[appeared[i]])
                    first_label_slice.append(k)
                for i in range(8):
                    rand = random.choice(file_pos[k])
                    while rand in appeared :
                        rand = random.choice(file_pos[k])
                    #On s'assure d'avoir que des images différentes dans query + dans gallery
                    appeared.append(rand)
                    sec_image_slice.append(file_image[appeared[i+2]])
                    sec_label_slice.append(k)

        return (first_image_slice, np.array(first_label_slice), sec_image_slice, np.array(sec_label_slice))
    if modal == "VtoT" :
        from_image = vis_img
        from_label = vis_id
        from_pos = vis_pos
        to_image = ir_img
        to_label = ir_id
        to_pos = ir_pos
    elif modal == "TtoV" :
        from_image = ir_img
        from_label = ir_id
        from_pos = ir_pos
        to_image = vis_img
        to_label = vis_id
        to_pos = vis_pos
        # Récupération des 20 dernier % d'images pour la phase de test
    for k in range(len(from_pos)):
        appeared = []
        # On chosiit deux personnes en query, le reste dans la gallery
        for j in range(2):
            rand = random.choice(from_pos[k])
            while rand in appeared:
                rand = random.choice(from_pos[k])
            appeared.append(rand)
            first_image_slice.append(from_image[appeared[j]])
            first_label_slice.append(k)
    for k in range(len(to_pos)):
        appeared = []
        for j in range(8):
            rand = random.choice(to_pos[k])
            while rand in appeared:
                rand = random.choice(to_pos[k])
            appeared.append(rand)
            sec_image_slice.append(from_image[appeared[j]])
            sec_label_slice.append(k)

    return (first_image_slice, np.array(first_label_slice), sec_image_slice, np.array(sec_label_slice))

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

# Print some of the images :
# print(trainset.train_color_image.shape)
# w=0
# for i in range(0, 250, 10):
#     w += 1
#     print(i)
#     plt.subplot(5,5,w)
#     plt.imshow(trainset.train_color_image[i])
# plt.show()

# testing set
# query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
# gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')


def process_test_single_sysu(data_path, method, trial=0, mode='all', relabel=False, reid="VtoT"):
    random.seed(trial)
    print("query")
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]


    files_query_visible = []
    files_gallery_visible = []
    files_query_thermal = []
    files_gallery_thermal = []
    for id in sorted(ids):
        #Selection of 1 img for gallery per cam and per id, the rest as query
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                rand = random.choice(new_files)
                files_gallery_visible.append(rand)
                for w in new_files:
                    if w != rand:
                        files_query_visible.append(w)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                rand = random.choice(new_files)
                files_gallery_thermal.append(rand)
                for w in new_files:
                    if w != rand:
                        files_query_thermal.append(w)
    query_img = []
    query_id = []
    query_cam = []
    gall_img = []
    gall_id = []
    gall_cam = []

    if reid == "VtoV":
        files_query = files_query_visible
        files_gallery = files_gallery_visible
    elif reid == "TtoT":
        files_query = files_query_thermal
        files_gallery = files_gallery_thermal

    for img_path in files_query:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    for img_path in files_gallery :
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam), gall_img, np.array(gall_id), np.array(gall_cam)


def process_query_sysu(data_path, method, trial=0, mode='all', relabel=False, reid="VtoT"):
    random.seed(trial)
    print("query")
    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)

    query_img = []
    query_id = []
    query_cam = []
    if reid=="VtoT" :
        files = files_rgb
    elif reid=="TtoV" :
        files = files_ir
    for img_path in files:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    #print(query_img)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, method, mode='all', trial=0, relabel=False, reid="VtoT"):
    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']
        ir_cameras = ['cam3', 'cam6']

    if method == "test":
        print("Test set called")
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    elif method == "valid":
        print("Validation set called")
        file_path = os.path.join(data_path, 'exp/val_id.txt')

    files_rgb = []
    files_ir = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]
    if reid in ["VtoT", "TtoV"]:
        for id in sorted(ids):
            for cam in rgb_cameras:
                img_dir = os.path.join(data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    files_rgb.append(random.choice(new_files))

            for cam in ir_cameras:
                img_dir = os.path.join(data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    files_ir.append(random.choice(new_files))
    elif reid in ["visible", "thermal"]:
        for id in sorted(ids):
            for cam in rgb_cameras:
                img_dir = os.path.join(data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    files_rgb.append(random.choice(new_files))
            for cam in ir_cameras:
                img_dir = os.path.join(data_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                    files_ir.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    if reid=="VtoT" :
        files = files_ir
    elif reid=="TtoV" :
        files = files_rgb
    for img_path in files:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)