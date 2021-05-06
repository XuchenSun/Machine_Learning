"""
@Class Name: classification
@Author: Xuchen Sun
@License: General Public License （GPL）
@Contact: xuchens@mun.ca
@Data: 2021-05-06
@Version: 1.0
@Hardware Note CPU:AMD3900
@Hardware Note GPU: EVGA GTX1080ti
@Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
@ML Model Type: CNN
@ML Model Type: Saved As Model_Layers.png
@Dataset: The  CIFAR10  Dataset
@The iterative part: Weight in CNN
"""

import Loading_Function_Set
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # enable XLA Devices

import Model_Set

import Train_Function_Set


EPOCHS=10

personal_loading_function_set=Loading_Function_Set.Personal_Loading_Function_Set
images_in_train_set,labels_in_train_set,images_in_test_set, labels_in_test_set=personal_loading_function_set.load_data_from_cifar10(self=personal_loading_function_set)


personal_model_set=Model_Set.Personal_Model_Set()
CNN_model=personal_model_set.build_CNN()

personal_function_set=Train_Function_Set.Personal_Train_Function_Set()
personal_function_set.train_CNN_model(CNN_model,images_in_train_set,labels_in_train_set,images_in_test_set, labels_in_test_set, epochs=EPOCHS)


loss_value_in_test_folder, accuracy_value_in_test_folder =CNN_model.evaluate(images_in_test_set,  labels_in_test_set, verbose=2)
print(accuracy_value_in_test_folder)