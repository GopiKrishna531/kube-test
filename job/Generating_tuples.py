'''
Here I'm to generate tuples to populate csv file. First I'll create csv files each for Original images
and adversarial images seperately and then If we want we can combine them.




'''
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
import json


IMAGENETTE_SAMPLE_DIR = '/scratch/scratch6/gopi/gopi/Imagenette_sample'
JSON_DEST_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/JsonFiles'
ADVERSARIAL_IMAGES_FGSM_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/AdversarialFGSM'
ADVERSARIAL_IMAGES_PGD_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/AdversarialPGD'
CLASS_ID_MAPPING_JSON_FILE_PATH = '/scratch/scratch6/gopi/gopi/CreatingDataset/Imagenet_class_id_mapping.json'
IS_ORIGINAL_GRADCAM_HEATMAPS_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/Heatmaps/GradCAM_Heatmaps/IS_Original_Heatmaps'
IS_ORIGINAL_ABLATIONCAM_HEATMAPS_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/Heatmaps/AblationCAM_Heatmaps/IS_Original_Heatmaps'
IS_GRADCAM_ADVERSARIAL_HEATMAPS_FGSM = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/Heatmaps/GradCAM_Heatmaps/IS_Adversarial_Heatmaps/IS_Adversarial_Heatmaps_FGSM'
IS_GRADCAM_ADVERSARIAL_HEATMAPS_PGD = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/Heatmaps/GradCAM_Heatmaps/IS_Adversarial_Heatmaps/IS_Adversarial_Heatmaps_PGD'

IS_ABLATIONCAM_ADVERSARIAL_HEATMAPS_FGSM = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/Heatmaps/AblationCAM_Heatmaps/IS_Adversarial_Heatmaps/IS_Adversarial_Heatmaps_FGSM'
IS_ABLATIONCAM_ADVERSARIAL_HEATMAPS_PGD = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/Heatmaps/AblationCAM_Heatmaps/IS_Adversarial_Heatmaps/IS_Adversarial_Heatmaps_PGD'

column_names = ['image_path','heatmap_path','logits','predicted_label','advornot','perturbation_method','perturbation_parameters','label_before_perturbation','xai_method']

'''
Explanation of column_names 

image_path -> It contains a path to the original image

heatmap_path -> It contains a path to heatmap image corresponding to the image in image_path 

logits -> It contains a 1000 length list containing the values before softmax layer of the model

predicted_label ->  The label predicted using the model for image in image_path
					It's a 3 tuple [class(str), class_id(integer), probability(float)]

advornot -> 0 or 1, where 0 means normal image and 1 means adversarial image.

perturbation method -> 0 or 1 -> 0 means FGSM and 1 means PGD.

perturbation parameters -> It's a 2 tuple list [epsilon, epochs]

label_before_perturbation -> means for an adversarial image we need to specify the label before perturbation 
							 It's a 3 tuple [class(str), class_id(integer), probability(float)]	

xai_method -> 0/1 where 0 is GradCAM and 1 is AblationCAM

'''

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

def create_new_logits_model(model):
    #https://stackoverflow.com/questions/58544097/turning-off-softmax-in-tensorflow-models

    assert model.layers[-1].activation == tf.keras.activations.softmax
    config = model.layers[-1].get_config()
    weights = [x.numpy() for x in model.layers[-1].weights]

    config['activation'] = tf.keras.activations.linear
    config['name'] = 'logits'

    new_layer = tf.keras.layers.Dense(**config)(model.layers[-2].output)
    new_model = tf.keras.Model(inputs=[self.model.input], outputs=[new_layer])
    new_model.layers[-1].set_weights(weights)

    assert new_model.layers[-1].activation == tf.keras.activations.linear
    return new_model

def get_logits(logits_model,image):

	pred_logits = logits_model.predict(image)[0]
	return pred_logits



def get_predicted_label(model, image):
	''' return a list with [class, class_id, probability]'''

    preds = model.predict(image)
    initial_class_id = np.argmax(preds[0])
    probability = preds[0][initial_class_id]
    initial_class = get_class_wrt_id(initial_class_id)
    return [initial_class,initial_class_id,probability]



def get_class_wrt_id(initial_class_id):
	with open(CLASS_ID_MAPPING_JSON_FILE_PATH,'r') as f:
  		data = json.load(f)
  	label = data[str(initial_class_id)]
  	return label
def get_perturbation_parameters(s,i):
	s = s.replace('for_','')
	if (i==0):
		return [float(s),1]
	else:
		eps=''
		epochs=''
		j=0
		for ch in s:
		  if (ch=='_'):
		    j=1
		  elif (j==0):
		    eps+=ch
		  else :
		    epochs+=ch
		return [float(eps),int(epochs)]

def get_label_before_perturbation(model,tail):
	sub_folders_list = os.listdir(IMAGENETTE_SAMPLE_DIR)
	fn = tail
	fn = fn.replace('adversarial_','')
	for each_sub_folder_name in sub_folders_list:
		file_path = IMAGENETTE_SAMPLE_DIR + f'/{each_sub_folder_name}/{fn}'
		if (os.path.exists(file_path)):
			break
	image = load_img(file_path, target_size=(224, 224))

    # create a batch and preprocess the image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return get_predicted_label(model,image)






if __name__ == '__main__':

	my_model = create_model()
	print('Finished creating model.')
	logits_model = create_new_logits_model(my_model)
	print('Finished creating logits model.')

	############################# We will work for Original images for both GradCAM and AblationCAM ################################

	print("######################## Started generating json tuples for original images ###################################")
	ALL_ORIGINAL_TUPLES=[]
	
  	all_files_list = []
  	

  	sub_folders_list = os.listdir(IMAGENETTE_SAMPLE_DIR)


  	for each_sub_folder_name in sub_folders_list:
      	sub_folder_path = IMAGENETTE_SAMPLE_DIR + f'/{each_sub_folder_name}/*.JPEG'

      	sub_folder_files = glob(sub_folder_path)
      	all_files_list.extend(sub_folder_files)

  	print('Extracted all files into a list, total files: ' + str(len(all_files_list)))


	for idx, each_input_image in enumerate(all_files_list):
		EACH_TUPLE=[]

	    print(f'Processing file: {idx}')

	    # (temp_path,ext) = os.path.splitext(each_input_image)
	    # (head,tail) = os.path.split(temp_path)

	    (head,tail) = os.path.split(each_input_image)
	    image = load_img(each_input_image, target_size=(224, 224))

	    # create a batch and preprocess the image
	    image = img_to_array(image)
	    image = np.expand_dims(image, axis=0)
	    image = imagenet_utils.preprocess_input(image)

		logits = get_logits(logits_model,image)

		EACH_TUPLE.append(each_input_image)
		
		gc_heatmap_path = f"{IS_ORIGINAL_GRADCAM_HEATMAPS_DIR}/{tail}"
		EACH_TUPLE.append(gc_heatmap_path)

		EACH_TUPLE.append(logits)

		EACH_TUPLE.append(get_predicted_label(my_model,image))

		EACH_TUPLE.append(0)

		EACH_TUPLE.append(None)
		EACH_TUPLE.append(None)
		EACH_TUPLE.append(None)
		EACH_TUPLE.append(0)

		ALL_ORIGINAL_TUPLES.append(EACH_TUPLE)

		# we just appended a tuple for GradCAM, we can add tuple for AblationCAM with just a slight modification
		# ac_heatmap_path = f"{IS_ORIGINAL_ABLATIONCAM_HEATMAPS_DIR}/{tail}"
		# EACH_TUPLE[1]=ac_heatmap_path
		# EACH_TUPLE[-1]=1
		# ALL_ORIGINAL_TUPLES.append(EACH_TUPLE)


	ALL_ORIGINAL_TUPLES_DICT = [dict(zip(column_names,each_list)) for each_list in ALL_ORIGINAL_TUPLES]


	# for each_image in all_images:
	#   result.append(dict(zip(Column_names, each_image)))

	original_dest = f"{JSON_DEST_DIR}/original_json.json"
	with open(original_dest, 'w') as f:
	  	json.dump(ALL_ORIGINAL_TUPLES_DICT, f)

	print('################################### Finished generating json tuples for Original images. #######################################')
	############################### For Adversarial images and GradCAM Heatmaps and with slightly adding code we can insert tuple for Ablation cam as well ###################################

	print('################################### Started generating json tuples for Adversarial images. ######################################')
	fgsm_pgd_source_dirs = [ADVERSARIAL_IMAGES_FGSM_DIR, ADVERSARIAL_IMAGES_PGD_DIR]
  	fgsm_pgd_gc_dest_dirs = [IS_GRADCAM_ADVERSARIAL_HEATMAPS_FGSM_DIR, IS_GRADCAM_ADVERSARIAL_HEATMAPS_PGD_DIR]
  	fgsm_pgd_ac_dest_dirs = [IS_ABLATIONCAM_ADVERSARIAL_HEATMAPS_FGSM_DIR, IS_ABLATIONCAM_ADVERSARIAL_HEATMAPS_PGD_DIR]

  	ALL_ADVERSARIAL_TUPLES =[]
  	i=0 # we will use i variable to note whether we are working for FGSM or PGD, 0 for FGSM and 1 for PGD
	for (each_source_path,each_gc_dest_path, each_ac_dest_path) in zip(fgsm_pgd_source_dirs, fgsm_pgd_gc_dest_dirs, fgsm_pgd_ac_dest_dirs):
		if (i==0):
			print('################# Started generating json tuples for FGSM images #########################')
		else :
			print('################# Started generating json tuples for PGD images #########################')

	    sub_folders_list = os.listdir(each_source_path)

	    for each_sub_folder_name in sub_folders_list:

	        sub_folder_path = each_source_path + f'/{each_sub_folder_name}/*.JPEG'

	        sub_folder_files = glob(sub_folder_path)
	        #all_files_list.extend(sub_folder_files)

	        #print('Extracted all files into a list, total files: ' + str(len(all_files_list)))

	        for idx, each_input_image in enumerate(sub_folder_files):
	        	EACH_TUPLE=[]

	            print(f'Processing file: {idx}')

	            (head,tail) = os.path.split(each_input_image)
			    image = load_img(each_input_image, target_size=(224, 224))

			    # create a batch and preprocess the image
			    image = img_to_array(image)
			    image = np.expand_dims(image, axis=0)
			    image = imagenet_utils.preprocess_input(image)


				logits = get_logits(logits_model,image)

				EACH_TUPLE.append(each_input_image)
				
				gc_heatmap_path = f"{each_gc_dest_path}/{each_sub_folder_name}/{tail}"
				EACH_TUPLE.append(gc_heatmap_path)

				EACH_TUPLE.append(logits)

				EACH_TUPLE.append(get_predicted_label(my_model,image))

				EACH_TUPLE.append(1)

				if(i==0):
					EACH_TUPLE.append(i)
					EACH_TUPLE.append(get_perturbation_parameters(each_sub_folder_name,i))

				else:
					EACH_TUPLE.append(i)
					EACH_TUPLE.append(get_perturbation_parameters(each_sub_folder_name,i))
				
				

				EACH_TUPLE.append(get_label_before_perturbation(my_model,tail))
				EACH_TUPLE.append(0)

				ALL_ADVERSARIAL_TUPLES.append(EACH_TUPLE)
				
				# we just appended a tuple for GradCAM, we can add tuple for AblationCAM with just a slight modification
				# ac_heatmap_path = f"{each_ac_dest_path}/{each_sub_folder_name}/{tail}"
				# EACH_TUPLE[1]=ac_heatmap_path
				# EACH_TUPLE[-1]=1
				
				# ALL_ADVERSARIAL_TUPLES.append(EACH_TUPLE)
			print(f"We have just completed generating json tuples for {each_sub_folder_name}")
    	print(f"We have just completed generating json tuples for {each_source_path}")
		if (i==0):
			print('################# Started generating json tuples for FGSM images #########################')
		else :
			print('################# Started generating json tuples for PGD images #########################')
		i+=1

	ALL_ADVERSARIAL_TUPLES_DICT = [dict(zip(column_names,each_list)) for each_list in ALL_ADVERSARIAL_TUPLES]
	# for each_image in all_images:
	#   result.append(dict(zip(Column_names, each_image)))

	adversarial_dest = f"{JSON_DEST_DIR}/adversarial_json.json"
	with open(adversarial_dest, 'w') as f:
	  	json.dump(ALL_ADVERSARIAL_TUPLES_DICT, f)
	print('################################### Finished generating json tuples for Adversarial images. ######################################')








		














