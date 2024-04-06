# define the model
model1 = make_yolov3_model()
 
# load the model weights
# I have loaded the pretrained weights in a separate dataset
weight_reader = WeightReader('/content/drive/MyDrive/yolov3.weights')
 
# set the model weights into the model
weight_reader.load_weights(model1)
 
# save the model to file
model1.save('model1.h5')
