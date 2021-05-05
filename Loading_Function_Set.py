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