from utils import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100
import numpy as np
import os

dest_dir = './res/cifar10/'

file_list = os.listdir(dest_dir)
file_lists = [file for file in file_list if file[-3:] =='npz']
print(file_lists)

for file in file_lists:
    a = np.load(dest_dir + file)
    for elem in a:
        print(elem)
        print(elem.shape)
    print(a.shape)
