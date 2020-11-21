import numpy as np
from PIL import Image
import torch.utils.data as data
import math
from torchvision import transforms
import matplotlib.pyplot as plt


class RegDBData_split(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, split="training" ):
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

        first80percent = int(len(color_img_file) * 80 / 100)
        first80percent -= int(len(color_img_file) * 80 / 100)%10

        if split == "training":
            for i in range(first80percent):
                if i % 10 < 7 :
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
            for i in range(first80percent):
                if i % 10 >= 7:
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


    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        if hasattr(self, "train_color_image"):
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
            img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        elif hasattr(self, "valid_color_image") :
            img1, target1 = self.valid_color_image[self.cIndex[index]], self.valid_color_label[self.cIndex[index]]
            img2, target2 = self.valid_thermal_image[self.tIndex[index]], self.valid_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class SYSUData_split(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None, split="training"):
        data_dir = '../Datasets/SYSU/'
        # Load training images (path) and labels
        #395 ids sont loadées sur les 491
        color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        color_pos, thermal_pos = GenIdx(color_label, thermal_label)
        _color_image = []
        _color_lab = []
        _thermal_image = []
        _thermal_lab = []
        SeventPercent = 0.7
        if split == "training" :
            #Dans chaque liste d'index d'une identité, on prends les 70% premieres images.
            for i in range(len(color_pos)):
                u = len(color_pos[i])
                for j in range(u) :
                    if u <= int(u*SeventPercent) :
                        _color_image.append(color_image[    color_pos[i][j]  ])
                        _color_lab.append(color_pos[i][j])
            for i in range(len(thermal_pos)):
                u = len(thermal_pos[i])
                for j in range(u):
                    if u <= int(u * SeventPercent) :
                        _thermal_image.append(color_image[thermal_pos[i][j]])
                        _thermal_lab.append(thermal_pos[i][j])
            # Labels
            self.train_color_label = _color_lab
            self.train_thermal_label = _thermal_lab
            # BGR to RGB
            self.train_color_image = _color_image
            self.train_thermal_image = _thermal_image
        if split == "validation" :
            #Dans chaque liste d'index d'une identité, on prends les 30% dernières images.
            print("in validation")
            for i in range(len(color_pos)):
                print("in boucle 1")
                u = len(color_pos[i])
                for j in range(u) :
                    print("in boucle 2")
                    if u > int(u*SeventPercent):
                        _color_image.append(color_image[    color_pos[i][j]  ])
                        _color_lab.append(color_pos[i][j])
            for i in range(len(thermal_pos)):
                u = len(thermal_pos[i])
                for j in range(u):
                    if u > int(u * SeventPercent):
                        _thermal_image.append(color_image[thermal_pos[i][j]])
                        _thermal_lab.append(thermal_pos[i][j])
            # Labels
            self.valid_color_label = _color_lab
            self.valid_thermal_label = _thermal_lab
            # BGR to RGB
            self.valid_color_image = _color_image
            self.valid_thermal_image = _thermal_image

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        if hasattr(self, "train_color_image"):
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
            img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        elif hasattr(self, "valid_color_image") :
            img1, target1 = self.valid_color_image[self.cIndex[index]], self.valid_color_label[self.cIndex[index]]
            img2, target2 = self.valid_thermal_image[self.tIndex[index]], self.valid_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class RegDBVisibleData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, split="training" ):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_1.txt'
        #Load color and thermal images + labels
        color_img_file, color_target = load_data(train_color_list)
        color_image = []
        color_lab = []

        #Get real and thermal images with good shape in a list
        # Training and valid are defined in the firsts 80s percent
        # Training is 70% and Valid is 30% of those 80s percent
        first80percent = int(len(color_img_file) * 80 / 100)
        first80percent -= int(len(color_img_file) * 80 / 100)%10
        if split == "training" :
            for i in range(first80percent):
                if i%10 < 7 :
                    img = Image.open(data_dir + color_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    color_image.append(pix_array)
                    color_lab.append(color_target[i])

            color_image = np.array(color_image)
            # Init color images / labels
            self.train_color_image = color_image
            self.train_color_label = color_lab

        if split == "validation" :
            for i in range(first80percent):
                if i%10 >= 7 :
                    img = Image.open(data_dir + color_img_file[i])
                    img = img.resize((144, 288), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    color_image.append(pix_array)
                    color_lab.append(color_target[i])

            color_image = np.array(color_image)

            # Init color images / labels
            self.valid_color_image = color_image
            self.valid_color_label = color_lab

        self.transform = transform
        # Prepare index
        self.cIndex = colorIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        if hasattr(self, "train_color_image"):
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        elif hasattr(self, "valid_color_image") :
            img1, target1 = self.valid_color_image[self.cIndex[index]], self.valid_color_label[self.cIndex[index]]

        img1 = self.transform(img1)

        return img1, target1
    def __len__(self):
        if hasattr(self, "train_color_image"):
            return len(self.train_color_label)
        elif hasattr(self, "valid_color_image"):
            return len(self.valid_color_label)

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

def process_test_regdb(img_dir, modal='visible', split = False):

    input_visible_data_path = img_dir + 'idx/train_visible_1.txt'
    input_thermal_data_path = img_dir + 'idx/train_thermal_1.txt'
    if modal == "visible" or modal == "VtoT" :
        with open(input_visible_data_path) as f:
            data_file_list = open(input_visible_data_path, 'rt').read().splitlines()
            # Get full list of image and labels
            file_image_visible = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
            file_label_visible = [int(s.split(' ')[1]) for s in data_file_list]
    if modal == "thermal" or modal == "VtoT":
        with open(input_thermal_data_path) as f:
            data_file_list = open(input_thermal_data_path, 'rt').read().splitlines()
            # Get full list of image and labels
            file_image_thermal = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
            file_label_thermal = [int(s.split(' ')[1]) for s in data_file_list]
    #If required, return half of the dataset in two slice
    if modal == "visible" :
        file_image = file_image_visible
        file_label = file_label_visible
    if modal == "thermal" :
        file_image = file_image_thermal
        file_label = file_label_thermal
    if modal == "thermal" or modal == "visible" :
        first_image_slice = []
        first_label_slice = []
        sec_image_slice = []
        sec_label_slice = []
        first80percent = int(len(file_image) * 80 / 100)
        first80percent -= int(len(file_image) * 80 / 100) % 10
        #Récupération des 20 dernier % d'images pour la phase de test
        for k in range(first80percent, len(file_image)):
            # On chosiit deux personnes en query, le reste dans la gallery
            if (k + 1) % 10 < 2 :
                first_image_slice.append(file_image[k])
                first_label_slice.append(file_label[k])
            else :
                sec_image_slice.append(file_image[k])
                sec_label_slice.append(file_label[k])
        return(first_image_slice, np.array(first_label_slice), sec_image_slice, np.array(sec_label_slice))
    elif modal == "VtoT" :
        first_image_slice = []
        first_label_slice = []
        sec_image_slice = []
        sec_label_slice = []
        first80percent = int(len(file_image_visible) * 80 / 100)
        first80percent -= int(len(file_image_visible) * 80 / 100) % 10
        #Récupération des 20 dernier % d'images pour la phase de test
        for k in range(first80percent, len(file_image_visible)):
            # On chosiit deux personnes en query, le reste dans la gallery
            if (k + 1) % 10 < 2 :
                #Query
                first_image_slice.append(file_image_visible[k])
                first_label_slice.append(file_label_visible[k])
            else :
                #Gallery
                sec_image_slice.append(file_image_thermal[k])
                sec_label_slice.append(file_label_thermal[k])
        return(first_image_slice, np.array(first_label_slice), sec_image_slice, np.array(sec_label_slice))


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