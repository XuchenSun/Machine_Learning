"""
@Class Name: Loading_Function_Set
@Class Aim: Facilitate using of loading function via Object Oriented Programming
@Author: Xuchen Sun
@License: General Public License （GPL）
@Contact: xuchens@mun.ca
@Data: 2021-05-05
@Version: 1.0
@Hardware Note CPU:AMD3900
@Hardware Note GPU: EVGA GTX1080ti
@Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
"""
import tensorflow as tf
from tensorflow.keras import datasets
class Personal_Loading_Function_Set:

  "Initialize objects of the class Personal_Loading_function_Set"
  def __init__(self):
    print("Build Personal Loading Function Set Successfully")
  def load_data_from_mnist(self):
    mnist = tf.keras.datasets.mnist
    (x_value_in_train_folder, y_value_in_train_folder), (x_value_in_test_folder, y_value_in_test_folder) = mnist.load_data()
    x_value_in_train_folder= x_value_in_train_folder / 255.0
    x_value_in_test_folder= x_value_in_test_folder / 255.0
    print("Load Data From Mnist Successfully")
    return x_value_in_train_folder,y_value_in_train_folder,x_value_in_test_folder,y_value_in_test_folder
  def load_data_from_cifar10(self):

    (images_in_train_set, labels_in_train_set), (images_in_test_set, labels_in_test_set) = datasets.cifar10.load_data()

    images_in_train_set = images_in_train_set / 255.0
    images_in_test_set = images_in_test_set / 255.0
    print("Load Data From Cifar10 Successfully")
    return images_in_train_set, labels_in_train_set,images_in_test_set, labels_in_test_set