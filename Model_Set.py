"""
@Class Name: Model_Set
@Class Aim: Facilitate using of building model function via Object Oriented Programming
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
class Personal_Model_Set:
  "Initialize objects of the class Personal_Model_set"
  def __init__(self):
    print("Build Personal Model Set Successfully")

  def build_FC(self):

    FC_model= tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(2048, activation='relu'),tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(512, activation='relu'),tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'),tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(16, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')# The output should be classified to from 1 to 10, therefore the dense should be 10
    ])
    print("Build FC With 14 Layers Successfully")
    tf.keras.utils.plot_model(FC_model, to_file='Model_Layers.png', show_shapes=True, show_layer_names=True,rankdir='TB', dpi=900, expand_nested=True)
    FC_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',# use crossentropy to calculate loss value
                  metrics=['accuracy'])
    return FC_model