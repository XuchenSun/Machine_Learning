"""
@Class Name: ML_Program_Set
@Author: Xuchen Sun
@License: General Public License （GPL）
@Contact: xuchens@mun.ca
@Data: 2021-05-05
@Version: 1.0
@Hardware Note CPU:AMD3900
@Hardware Note GPU: EVGA GTX1080ti
@Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
"""
import Loading_Function_Set
import Model_Set
import Train_Function_Set
import os

class Running_ML_Program_Set:
    def __init__(self):
        print("Running_ML_Program_Set  Initialized")
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # enable XLA Devices
    def running_ML_program_classification(self):

        EPOCHE=10

        "Use function in class to build FC model"
        personal_model_set=Model_Set.Personal_Model_Set()
        FC_model=personal_model_set.build_FC()

        "Use function in class to load data from MNIST"
        personal_loading_function_set=Loading_Function_Set.Personal_Loading_Function_Set()
        x_in_train,y_in_train,x_in_test,y_in_test=personal_loading_function_set.load_data_from_mnist()


        "Use the data to train the model "
        train_set=Train_Function_Set.Personal_Train_Function_Set()
        train_set.train_FC_model(FC_model,x_in_train,y_in_train,x_in_test,y_in_test,EPOCHE)



