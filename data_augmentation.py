import numpy as np
from skimage.util import random_noise

def data_aug(visible_images = None, Thermal_images = None, Visible_labels = None, Thermal_labels= None) :

    final_visible_data = []
    final_visible_target = []
    final_thermal_data = []
    final_thermal_target = []
    # print(trainset.train_color_label)
    if visible_images is not None :
        visible_img_number = visible_images.shape[0]
        for k in range(visible_img_number):
            final_visible_data.append(visible_images[k])
            final_visible_data.append(np.fliplr(visible_images[k]))
            final_visible_data.append(random_noise(visible_images[k], var=0.01))
            for j in range(3):
                final_visible_target.append(Visible_labels[k])
    if Thermal_images is not None :
        thermal_img_number = Thermal_labels.shape[0]
        for k in range(thermal_img_number):
            final_thermal_data.append(visible_images[k])
            final_thermal_data.append(np.fliplr(visible_images[k]))
            final_thermal_data.append(random_noise(visible_images[k], var=0.01))
            for j in range(3):
                final_thermal_target.append(Visible_labels[k])

    return np.array(final_visible_data), np.array(final_visible_target),\
           np.array(final_thermal_data), np.array(final_thermal_target)