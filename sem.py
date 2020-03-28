""" SEM small dataset loading function
"""

import numpy as np
import os
from PIL import Image

def get_num_steps(path, step=10):
    images = os.listdir(path)
    num_images = len(images)
    image = images[1]

    im = Image.open(path + image)
    imarray = np.array(im)

    im_shape = imarray.shape
    # # The length of the steps
    x_step_norm = np.floor((im_shape[0] - 32) / step) + 1
    y_step_norm = np.floor((im_shape[1] - 32) / step) + 1
    print(x_step_norm, ' : ',  y_step_norm)
    num_steps = x_step_norm * y_step_norm
    return num_steps, num_images


# sliding window to get the pictures
def load_images(path, num_steps, num_images, step=10, if_test=False):
    images = os.listdir(path)
    x_ = np.empty((int(num_steps * num_images), 32, 32, 1), dtype='uint8')
    y_ = np.zeros((int(num_steps * num_images)), dtype='uint8')
    labels = np.zeros((int(num_steps * num_images), 2), dtype='uint8')
    idx = 0
    for image_num, image_name in enumerate(images):
        image = np.array(Image.open(path + image_name))
        # give an empty assignment
        gt = []
        if if_test:
            image_id = image_name[:-4] + '_gt.png'
            gt = np.array(Image.open('./dataset/Anomalous/gt/' + image_id))
        # initial the sliding window location
        h = 0
        w = 0
        for num in range(int(num_steps)):
            if h + 32 < image.shape[0]:
                h_real = h
                w_real = w
                h += step
            else:
                if w + 32 < image.shape[1]:
                    h_real = image.shape[0] - 33
                    w_real = w
                    h = 0
                    w += step
                else:
                    w = image.shape[1] - 33
                    h_real = h
                    w_real = w
                    h += step
            x_[idx, :, :, :] = np.expand_dims(image[h_real:h_real + 32, w_real:w_real + 32], axis=2)
            labels[idx, :] = [image_num, num]
            # im = Image.fromarray(x_[idx, :, :])
            # im.save('./test/' + image_name + '_' + num + '.jpg')
            if if_test:
                # we can not use area.max since it would return a pointer to the memory
                # instead, np.amax() function is useful.
                if np.amax(gt[h_real:h_real + 32, w_real:w_real + 32]):
                    y_[idx] = 1
            idx += 1
    return x_, y_, labels


def load_data(step=10, if_saved=1, saved_path='./mid_data/'):
    """
    load local SEM data
    You should define its steps (The same step length of horizontal and vertical should be more reasonable)
    :return:
    (X_train, Y_train), (X_test, Y_test)
    actually there`s only two kind of the label
    """
    if if_saved == 1:
        x_train = np.load(saved_path + 'x_train.npy')
        y_train = np.load(saved_path + 'y_train.npy')
        x_test = np.load(saved_path + 'x_test.npy')
        y_test = np.load(saved_path + 'y_test.npy')
        test_label = np.load(saved_path + 'test_label.npy')
        print("load mid data successfully!")

        return (x_train, y_train), (x_test, y_test, test_label)

    # # How to reconstruct an image with the results => we can do it later

    normal_path = './dataset/Normal/'
    anomalous_path = './dataset/Anomalous/images/'
    # anomalous_gt = './dataset/Anomalous/gt/'
    # # Create the empty
    num_train_steps, num_train_images = get_num_steps(normal_path)
    x_train, y_train, _ = load_images(normal_path, num_train_steps, num_train_images)

    num_test_steps, num_test_images = get_num_steps(anomalous_path)
    x_test, y_test, test_label = load_images(anomalous_path, num_test_steps, num_test_images, step=10, if_test=True)

    # 把 y 和 test_label 放到一起， 或者多一个参数
    if if_saved == 2:
        np.save(saved_path + 'x_train.npy', x_train)
        np.save(saved_path + 'y_train.npy', y_train)
        np.save(saved_path + 'x_test.npy', x_test)
        np.save(saved_path + 'y_test.npy', y_test)
        np.save(saved_path + 'test_label.npy', test_label)
        print("save mid data successfully!")

    return (x_train, y_train), (x_test, y_test, test_label)

def create_image(img, save_path, img_name):
    im_shape = img.shape
    image = np.zeros([670, 1000], dtype='uint8')
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            image[i*10:i*10+10, j*10:j*10+10] = int(img[i, j]*255)
    im = Image.fromarray(image)
    im.save(save_path + img_name + '.png')

    print(save_path + img_name + '.png !')