from utils import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100

(x_train, y_train), (x_test, y_test) = load_cifar10()

x_train_task = x_train[y_train.flatten() == single_class_ind]


print(y_train.flatten())
print('------------------')
print(x_train.shape[1:])
print('------------------')
print(x_train[1:])
