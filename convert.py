import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow.lite as lite

#print(tf.version)
# #from tensorflow.contrib import lite
# converter = tf.lite.TFLiteConverter.from_keras_model_file( 'model.h5' ) # Your model's name
# model = converter.convert()
# file = open( 'model.tflite' , 'wb' )
# file.write( model )

# Convert the model.
# f = open('RGModel.h5','rb')
model=tf.keras.models.load_model('RGModel.h5')
converter = lite.TFLiteConverter.from_keras_model_file(model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
# converter=tf.compat.v1.lite.TFLiteConverter(model)
# tflite_model = converter.convert()
#
# with open('RGmodel.tflite', 'wb') as f:
#   f.write(tflite_model)
#converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('RGModel.h5')
#tflite_model = converter.convert()

# Save the model.
#with open('RGmodel.tflite', 'wb') as f:
  #f.write(tflite_model)

# from tensorflow.contrib import lite
# converter = lite.TFLiteConverter.from_keras_model_file('RGmodel.h5')
# tfmodel = converter.convert()
# open ("model.tflite" , "wb") .write(tfmodel)





# converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('RGModel.h5')
# tflite_model = converter.convert()
#
# # Save the model.
# with open('RGModel.tflite', 'wb') as f:
#   f.write(tflite_model)







#f = open("RGModel.hdf5", "r")
# converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("RGModel.hdf5")
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)
