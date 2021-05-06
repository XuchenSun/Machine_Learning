"""
@Class Name: Train_Function_Set
@Class Aim: Facilitate using of training function via Object Oriented Programming
@Author: Xuchen Sun
@License: General Public License （GPL）
@Contact: xuchens@mun.ca
@Data: 2021-05-05
@Version: 1.0
@Hardware Note CPU:AMD3900
@Hardware Note GPU: EVGA GTX1080ti
@Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
"""

class Personal_Train_Function_Set:
  "Initialize objects of the class Personal_Train_set"
  def __init__(self):

    print("Build Personal Train Set Successfully")


  def train_FC_model(self,model, x_value_in_train_folder, y_value_in_train_folder, x_value_in_test_folder, y_value_in_test_folder,epoche):
    print("FC model is training")
    model.fit(x_value_in_train_folder, y_value_in_train_folder, epochs=epoche)
    model.evaluate(x_value_in_test_folder, y_value_in_test_folder, verbose=2)

  def train_CNN_model(self,model,images_in_train_set,labels_in_train_set,images_in_test_set, labels_in_test_set,epochs):
    print("CNN model model is training")
    model.fit(images_in_train_set, labels_in_train_set, epochs=epochs,validation_data=(images_in_test_set, labels_in_test_set))
    model.evaluate(images_in_test_set,labels_in_test_set, verbose=2)
