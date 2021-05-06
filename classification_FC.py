"""
@Class Name: classification
@Author: Xuchen Sun
@License: General Public License （GPL）
@Contact: xuchens@mun.ca
@Data: 2021-05-05
@Version: 1.0
@Hardware Note CPU:AMD3900
@Hardware Note GPU: EVGA GTX1080ti
@Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
@ML Model Type: Dense Layers(fully connected layers)
@ML Model Type: Saved As Model_Layers.png
@Dataset: The Mnist Dataset
@The iterative part: Weight in Fully Connected Layers
"""
import ML_Program_Set

"Running classification in ML by just two command via OOP"
program_set=ML_Program_Set.Running_ML_Program_Set()
program_set.running_ML_program_classification_FC()
