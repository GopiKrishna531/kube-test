#from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16,vgg16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
#from keras.utils import plot_model
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
from glob import glob
import os
import cv2


IMAGENETTE_SAMPLE_DIR = '/scratch/scratch6/gopi/gopi/Imagenette_sample'
ADVERSARIAL_IMAGES_FGSM_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/AdversarialFGSM'
ADVERSARIAL_IMAGES_PGD_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/AdversarialPGD'
IS_GRADCAM_ORIGINAL_HEATMAPS_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/GradCAM_Heatmaps/IS_Original_Heatmaps'
IS_GRADCAM_ADVERSARIAL_HEATMAPS_FGSM_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/GradCAM_Heatmaps/IS_Adversarial_Heatmaps/IS_Adversarial_Heatmaps_FGSM'
IS_GRADCAM_ADVERSARIAL_HEATMAPS_PGD_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/GradCAM_Heatmaps/IS_Adversarial_Heatmaps/IS_Adversarial_Heatmaps_PGD'



def create_model():
  # load the pre-trained CNN from disk
  my_model = VGG16(weights="imagenet")
  #my_model.summary()
  #plot_model(my_model,to_file='vgg16_model.png',show_shapes=False,show_layer_names=True)


  my_model.trainable=False

  # my_model = load_model('/content/VGG16_VOC2012_model.h5')
  # my_model.summary()
  # plot_model(my_model)
  return my_model


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output.shape) == 4:
                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(inputs=[self.model.inputs], outputs= [self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            print("convOutputs dimensions :",convOutputs.shape)
            print("predictions dimensions :",predictions.shape)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        print("weights shape :",weights.shape)
        print("convOutputs shape :",convOutputs.shape)
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

def build_folders():
    paths_list = [VAL_PERTURBATIONS_DIR, VAL_ADVERSARIAL_DIR]
    for each_path in paths_list:
        for eps in EPS_LIST:
            os.mkdir(os.path.join(each_path, f'for_{eps}'))


def save_image(x,filename,images_dir=None):
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,2]
    t[:,:,1] = x[0][:,:,1]
    t[:,:,2] = x[0][:,:,0]  
    t=np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255
    #plt.imsave(filename,t)
    plt.imsave(f"{images_dir}/{filename}",t)

if __name__ == '__main__':

  all_files_list = []
  # build_folders()
  # print('Finished creating sub-folders.')

  sub_folders_list = os.listdir(IMAGENETTE_SAMPLE_DIR)

  for each_sub_folder_name in sub_folders_list:
      sub_folder_path = IMAGENETTE_SAMPLE_DIR + f'/{each_sub_folder_name}/*.JPEG'

      sub_folder_files = glob(sub_folder_path)
      all_files_list.extend(sub_folder_files)

  print('Extracted all files into a list, total files: ' + str(len(all_files_list)))

  my_model = create_model()
  print('Finished creating model.')

for idx, each_input_image in enumerate(all_files_list):

    print(f'Processing file: {idx}')

    # (temp_path,ext) = os.path.splitext(each_input_image)
    # (head,tail) = os.path.split(temp_path)


    

    (head,tail) = os.path.split(each_input_image)
    image = load_img(each_input_image, target_size=(224, 224))

    # create a batch and preprocess the image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    preds = my_model.predict(image)
    initial_class = np.argmax(preds[0])



    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(my_model, initial_class)
    heatmap = cam.compute_heatmap(image)


    save_image(heatmap,f"{tail}",IS_GRADCAM_ORIGINAL_HEATMAPS_DIR)