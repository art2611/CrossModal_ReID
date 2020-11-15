import numpy as np
from PIL import Image
import torch.utils.data as data
import math
from torchvision import transforms
import matplotlib.pyplot as plt


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'
        #Load color and thermal images + labels
        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        #Get real and thermal images with good shape in a list
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)

        train_color_image = np.array(train_color_image)
        train_thermal_image = np.array(train_thermal_image)

        # Init color images / labels
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # Init themal images / labels
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform

        # Prepare index
        self.cIndex = colorIndex
        self.tIndex = thermalIndex


    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)



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
        # Training => return 50%
        if split == "training" :
            first50percent = int(len(color_img_file)*50/100)
            for i in range(first50percent):
                img = Image.open(data_dir + color_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                color_image.append(pix_array)
                color_lab.append(color_target[i])

            for i in range(first50percent):
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
            first50percent = int(len(color_img_file) * 50 / 100)
            first80percent = int(len(color_img_file) * 80 / 100)
            for i in range(first50percent , first80percent):
                img = Image.open(data_dir + color_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                color_image.append(pix_array)
                color_lab.append(color_target[i])

            for i in range(first50percent , first80percent):
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
        if split == "testing" :
            first80percent = int(len(color_img_file)*80/100)
            print(f' last 20% {len(color_img_file) - (first80percent +1)}')
            for i in range(first80percent, len(color_img_file)):
                img = Image.open(data_dir + color_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                color_image.append(pix_array)
                color_lab.append(color_target[i])

            for i in range(first80percent +1, len(color_img_file)):
                img = Image.open(data_dir + thermal_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                thermal_image.append(pix_array)
                thermal_lab.append(thermal_target[i])
            color_image = np.array(color_image)
            thermal_image = np.array(thermal_image)
            # Init color images / labels
            self.test_color_image = color_image
            self.test_color_label = color_lab

            # Init themal images / labels
            self.test_thermal_image = thermal_image
            self.test_thermal_label = thermal_lab

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
        elif hasattr(self, "test_color_image") :
            img1, target1 = self.test_color_image[self.cIndex[index]], self.test_color_label[self.cIndex[index]]
            img2, target2 = self.test_thermal_image[self.tIndex[index]], self.test_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBVisibleData_split(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, split="training" ):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list = data_dir + 'idx/train_visible_1.txt'
        #Load color and thermal images + labels
        color_img_file, color_target = load_data(train_color_list)
        color_image = []
        color_lab = []

        #Get real and thermal images with good shape in a list
        # Training => return 50%
        if split == "training" :
            first50percent = int(len(color_img_file)*50/100)
            for i in range(first50percent):
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
            first50percent = int(len(color_img_file) * 50 / 100)
            first80percent = int(len(color_img_file) * 80 / 100)
            for i in range(first50percent , first80percent):
                img = Image.open(data_dir + color_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                color_image.append(pix_array)
                color_lab.append(color_target[i])

            color_image = np.array(color_image)

            # Init color images / labels
            self.valid_color_image = color_image
            self.valid_color_label = color_lab

        if split == "testing" :
            first80percent = int(len(color_img_file)*80/100)
            print(f' last 20% {len(color_img_file) - (first80percent +1)}')
            for i in range(first80percent, len(color_img_file)):
                img = Image.open(data_dir + color_img_file[i])
                img = img.resize((144, 288), Image.ANTIALIAS)
                pix_array = np.array(img)
                color_image.append(pix_array)
                color_lab.append(color_target[i])

            color_image = np.array(color_image)

            # Init color images / labels
            self.test_color_image = color_image
            self.test_color_label = color_lab


        self.transform = transform
        # Prepare index
        self.cIndex = colorIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
        if hasattr(self, "train_color_image"):
            img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        elif hasattr(self, "valid_color_image") :
            img1, target1 = self.valid_color_image[self.cIndex[index]], self.valid_color_label[self.cIndex[index]]
        elif hasattr(self, "test_color_image") :
            img1, target1 = self.test_color_image[self.cIndex[index]], self.test_color_label[self.cIndex[index]]

        img1 = self.transform(img1)

        return img1, target1
    def __len__(self):
        if hasattr(self, "train_color_image"):
            return len(self.train_color_label)
        elif hasattr(self, "valid_color_image"):
            return len(self.valid_color_label)
        elif hasattr(self, "test_color_image"):
            return len(self.test_color_label)

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

def process_test_regdb(img_dir, trial=1, modal='visible', split = False):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/train_visible_1.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/train_thermal_1.txt'

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
    #If required, return half of the dataset in two slice
    if split :
        first_image_slice = []
        first_label_slice = []
        sec_image_slice = []
        sec_label_slice = []
        first80percent = int(len(file_image) * 80 / 100)
        #Récupération des 20 dernier % d'images pour la phase de test
        for k in range(first80percent, len(file_image)):
            if trial == 1 :
                w = ((k+1)%2 == 0)
            elif trial == 2 :
                w = ((k+1)%5 < 5)
            elif trial == 3 :
                w = ((k+1)%4 < 4)
            elif trial == 4 :
                w = ((k+1)%6 <6)
            if w :
            # if (k+1)%10 < 5 :
                first_image_slice.append(file_image[k])
                first_label_slice.append(file_label[k])
            else :
                sec_image_slice.append(file_image[k])
                sec_label_slice.append(file_label[k])
        return(first_image_slice, np.array(first_label_slice), sec_image_slice, np.array(sec_label_slice))
    return file_image, np.array(file_label)

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