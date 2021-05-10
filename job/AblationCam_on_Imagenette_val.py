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
import time
import warnings
warnings.filterwarnings("ignore")


IMAGENETTE_SAMPLE_DIR = '/scratch/scratch6/gopi/gopi/imagenette2-320/val'
ADVERSARIAL_IMAGES_FGSM_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_val/AdversarialFGSM'
ADVERSARIAL_IMAGES_PGD_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_val/AdversarialPGD'
ORIGINAL_ABLATIONCAM_HEATMAPS_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_val/Heatmaps/AblationCAM_Heatmaps/Original_Heatmaps'
ABLATIONCAM_ADVERSARIAL_HEATMAPS_FGSM_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_val/Heatmaps/AblationCAM_Heatmaps/Adversarial_Heatmaps/Adversarial_Heatmaps_FGSM'
ABLATIONCAM_ADVERSARIAL_HEATMAPS_PGD_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_val/Heatmaps/AblationCAM_Heatmaps/Adversarial_Heatmaps/Adversarial_Heatmaps_PGD'



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

class AblationCAM:
    def __init__(self, model, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = None
        self.layerName = layerName
        self.new_model=self.get_new_model()
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
        self.classifier_layer_names=self.get_classifier_layer_names()
        self.last_conv_layer=self.model.get_layer(self.layerName)
        self.last_conv_layer_model=self.get_last_conv_layer_model()
        self.classifier_model=self.get_classifier_model()
        
    def get_classIdx(self, classId):
        self.classIdx = classId
    
    def get_new_model(self):
        #https://stackoverflow.com/questions/58544097/turning-off-softmax-in-tensorflow-models

        assert self.model.layers[-1].activation == tf.keras.activations.softmax
        config = self.model.layers[-1].get_config()
        weights = [x.numpy() for x in self.model.layers[-1].weights]

        config['activation'] = tf.keras.activations.linear
        config['name'] = 'logits'

        new_layer = tf.keras.layers.Dense(**config)(self.model.layers[-2].output)
        new_model = tf.keras.Model(inputs=[self.model.input], outputs=[new_layer])
        new_model.layers[-1].set_weights(weights)

        assert new_model.layers[-1].activation == tf.keras.activations.linear
        return new_model

    def get_classifier_layer_names(self):
        classifier_lay_names=[]
        for layer in reversed(self.new_model.layers):
              # check to see if the layer has a 4D output
              if len(layer.output.shape) != 4:
                classifier_lay_names[:0]=[layer.name]
              else:
                break
        return classifier_lay_names

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

    def get_last_conv_layer_model(self):
        #last_conv_layer = self.model.get_layer(layerName)
        last_conv_layer_model = Model(self.model.inputs, self.last_conv_layer.output)
        return last_conv_layer_model

    def get_classifier_model(self):
        classifier_input = tf.keras.Input(shape=self.last_conv_layer.output.shape[1:])
        x = classifier_input

        for layer_name in self.classifier_layer_names:
          x = self.new_model.get_layer(layer_name)(x)
        classifier_model = Model(classifier_input, x)
        return classifier_model

    
    
    def compute_heatmap(self, image, eps=1e-8):
        #activations = self.last_conv_layer_model.predict(image) 
        #print(activations.shape)
        #print(activations[-1].shape)
        #last_layer_activation = activations[-1]
        #last_layer_activation = np.expand_dims(last_layer_activation,axis=0)
        #print(last_layer_activation.shape)

        # heatmap = np.mean(last_layer_activation, axis=-1)
        # heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        values = []
        activations = self.last_conv_layer_model(image) 
        last_layer_activation = activations[-1]
        activation = last_layer_activation
        activation = np.expand_dims(activation,axis=0)
        for i in range(last_layer_activation.shape[-1]):
            activation_copy = activation.copy()
            activation_copy[0,:,:,i]=0.
            prediction = self.classifier_model(activation_copy)
            a= prediction[0][self.classIdx]
            values.append(a.numpy())
        #print(np.sum(values))


        pred_prob = self.new_model(image)[0][self.classIdx]
        pred_probabilities = np.array([pred_prob]*last_layer_activation.shape[-1])
        values=np.array(values)
        weight_ratio=(pred_probabilities-values)/pred_probabilities
        #weight_ratio

        ab_cam = tf.reduce_sum(tf.multiply(weight_ratio, last_layer_activation,), axis=-1)
        #print(ab_cam.shape)
        #print(ab_cam)
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(ab_cam.numpy(), (w, h))
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

def save_heatmap(x,filename,images_dir=None):
    plt.imsave(f"{images_dir}/{filename}",x)

# This below main, I used for creating Heatmaps for Original images of Imagenette sample

# if __name__ == '__main__':

#     all_files_list = []
#     # build_folders()
#     # print('Finished creating sub-folders.')

#     sub_folders_list = os.listdir(IMAGENETTE_SAMPLE_DIR)

#     for each_sub_folder_name in sub_folders_list:
#       sub_folder_path = IMAGENETTE_SAMPLE_DIR + f'/{each_sub_folder_name}/*.JPEG'

#       sub_folder_files = glob(sub_folder_path)
#       all_files_list.extend(sub_folder_files)

#     print('Extracted all files into a list, total files: ' + str(len(all_files_list)))

#     my_model = create_model()
#     print('Finished creating model.')

#     cam = AblationCAM(my_model)

#     for idx, each_input_image in enumerate(all_files_list):

#         print(f'Processing file: {idx}')

#         # (temp_path,ext) = os.path.splitext(each_input_image)
#         # (head,tail) = os.path.split(temp_path)


        

#         (head,tail) = os.path.split(each_input_image)
#         image = load_img(each_input_image, target_size=(224, 224))

#         # create a batch and preprocess the image
#         image = img_to_array(image)
#         image = np.expand_dims(image, axis=0)
#         image = imagenet_utils.preprocess_input(image)

#         preds = my_model(image)
#         initial_class = np.argmax(preds[0])



#         # initialize our gradient class activation map and build the heatmap
#         #start = time.time()
#         #cam = AblationCAM(my_model, initial_class)
#         #end = time.time()
#         #print(f"For creating cam object it takes : {round(start-end,2)} sec")
#         #start = time.time()

#         cam.get_classIdx(initial_class)
#         heatmap = cam.compute_heatmap(image)
#         #end = time.time()
#         #print(f"compute_heatmap takes : {round(start-end,2)} sec")


#         save_heatmap(heatmap,f"{tail}",ORIGINAL_ABLATIONCAM_HEATMAPS_DIR)

#     print("########################################## Completed generating heatmaps for Original images ###################################################")

# This below main, I used for creating Heatmaps for Adversarial images(FGSM,PGD) of Imagenette sample

if __name__ == '__main__':

    #all_files_list = []
    # build_folders()
    # print('Finished creating sub-folders.')
    fgsm_pgd_source_dirs = [ADVERSARIAL_IMAGES_FGSM_DIR, ADVERSARIAL_IMAGES_PGD_DIR]
    fgsm_pgd_dest_dirs = [ABLATIONCAM_ADVERSARIAL_HEATMAPS_FGSM_DIR, ABLATIONCAM_ADVERSARIAL_HEATMAPS_PGD_DIR]

    my_model = create_model()
    print('Finished creating model.')

    cam = AblationCAM(my_model)

    for (each_source_path,each_dest_path) in zip(fgsm_pgd_source_dirs, fgsm_pgd_dest_dirs):

        print(f"####################################### Started generating heatmaps for {each_source_path} #########################################")

        sub_folders_list = os.listdir(each_source_path)

        for each_sub_folder_name in sub_folders_list:
            print(f"########################### Started generating heatmaps for {each_sub_folder_name} ###############################")
            sub_folder_path = each_source_path + f'/{each_sub_folder_name}/*.JPEG'

            sub_folder_files = glob(sub_folder_path)
            #all_files_list.extend(sub_folder_files)

            #print('Extracted all files into a list, total files: ' + str(len(all_files_list)))



            for idx, each_input_image in enumerate(sub_folder_files):

                print(f'Processing file: {idx}')

                # (temp_path,ext) = os.path.splitext(each_input_image)
                # (head,tail) = os.path.split(temp_path)


                

                (head,tail) = os.path.split(each_input_image)
                image = load_img(each_input_image, target_size=(224, 224))

                # create a batch and preprocess the image
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)

                preds = my_model(image)
                initial_class = np.argmax(preds[0])



                # initialize our gradient class activation map and build the heatmap

                #cam = AblationCAM(my_model, initial_class)
                cam.get_classIdx(initial_class)
                heatmap = cam.compute_heatmap(image)

                dest = f"{each_dest_path}/{each_sub_folder_name}"
                save_heatmap(heatmap,f"{tail}",dest)
            print(f"##################################### Completed generating heatmaps for {each_sub_folder_name} #####################################")
        print(f"####################################### completed generating heatmaps for {each_source_path} #########################################")
    print("########################################## Completed generating heatmaps for Adversarial images ###################################################")



