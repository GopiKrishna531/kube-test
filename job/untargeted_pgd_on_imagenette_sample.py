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

IMAGENETTE_DIR_PATH = '/scratch/scratch6/gopi/gopi/imagenette2-320/val'

TRAIN_PERTURBATIONS_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/train/PerturbationsPGD'
TRAIN_ADVERSARIAL_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/train/AdversarialPGD'

VAL_PERTURBATIONS_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/PerturbationsPGD'
VAL_ADVERSARIAL_DIR = '/scratch/scratch6/gopi/gopi/CreatingDataset/ImagenetteDataset_sample/val/AdversarialPGD'



# EPS_LIST = [0.01,0.05,0.1,0.5]
# EPOCHS_LIST = [100,200,300]
EPS_LIST = [0.01,0.1]
EPOCHS_LIST = [200]
loss_object = tf.keras.losses.CategoricalCrossentropy()

def build_folders():
    paths_list = [VAL_PERTURBATIONS_DIR, VAL_ADVERSARIAL_DIR]
    for each_path in paths_list:
        for eps in EPS_LIST:
            for epoch in EPOCHS_LIST:
                os.mkdir(os.path.join(each_path, f'for_{eps}_{epoch}'))


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


def predict_label(image, my_model):
  
  preds=my_model.predict(image)
  i=np.argmax(preds[0])
  decoded = imagenet_utils.decode_predictions(preds)
  (imagenetID, label, prob) = decoded[0][0]
  res=(i,label,prob*100)
  label = "{} and {}: {:.2f}%".format(i,label, prob * 100)
  print("[INFO] {}".format(label))
  return res


def predict_label_my_model(image, my_model):
  preds=my_model.predict(image)
  i = np.argmax(preds[0])
  prob = preds[0][i]
  label = classes[i]
  label = "{} and {}: {:.2f}%".format(i,label, prob * 100)
  print("[INFO] {}".format(label))


# Inverse of the preprocessing and plot the image
def plot_img(x):
    """
    x is a BGR image with shape (? ,224, 224, 3) 
    """
    # preds = model.predict(x)
    # print('predicted :',imagenet_utils.decode_predictions(preds,top=3)[0])
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,2]
    t[:,:,1] = x[0][:,:,1]
    t[:,:,2] = x[0][:,:,0]  
    t=np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255
    plt.imshow(t) #+[123.68, 116.779, 103.939]
    plt.grid('off')
    plt.axis('off')
    plt.show()

def save_image(x,filename,images_dir=None):
    t = np.zeros_like(x[0])
    t[:,:,0] = x[0][:,:,2]
    t[:,:,1] = x[0][:,:,1]
    t[:,:,2] = x[0][:,:,0]  
    t=np.clip((t+[123.68, 116.779, 103.939]), 0, 255)/255
    #plt.imsave(filename,t)
    plt.imsave(f"{images_dir}/{filename}",t)


def create_adversarial_pattern_pgd(input_image, input_label, my_model, epochs_list=[200], eps=0.01):
  perturbations = np.zeros_like(input_image)
  adv_image = input_image
  #prev_probs=[]
  j=0
  perturbations_per_epochs = []
  for i in range(np.max(epochs_list)):

    with tf.GradientTape() as tape:
      tape.watch(adv_image)
      prediction = my_model(adv_image)
      loss = loss_object(input_label, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient=tape.gradient(loss,adv_image)
    delta=tf.sign(gradient[0])
    perturbations=perturbations+delta
    adv_image=input_image+eps*perturbations

    if(j<len(epochs_list) and i==epochs_list[j]-1):
      j++
      perturbations_per_epochs.append(perturbations)



    # if(i%20==0):
    #   predict_label(adv_image,my_model)

    
  return perturbations_per_epochs


if __name__ == '__main__':
    print('Started execution....!!')

    all_files_list = []
    build_folders()
    print('Finished creating sub-folders.')

    my_model = create_model()
    print('Finished creating model.')

    sub_folders_list = os.listdir(IMAGENETTE_DIR_PATH)

    for each_sub_folder_name in sub_folders_list:
        sub_folder_path = IMAGENETTE_DIR_PATH + f'/{each_sub_folder_name}/*.JPEG'

        sub_folder_files = glob(sub_folder_path)
        all_files_list.extend(sub_folder_files)

    print('Extracted all files into a list, total files: ' + str(len(all_files_list)))

    # for epochs in EPOCHS_LIST:
    #   print(f"We are working for epochs = {epochs}")

      
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

        # Get the input label of the image.
        class_index = initial_class 
        label = tf.one_hot(class_index, preds.shape[-1])
        label = tf.reshape(label, (1, preds.shape[-1]))
        #print(label.shape)
        #print(label[0][initial_class])
        perturbations_per_epochs = create_adversarial_pattern_pgd(tf.convert_to_tensor(image, dtype=tf.float32),label, my_model, EPOCHS_LIST,eps)
        
        i=0
        for epochs in EPOCHS_LIST:
          for eps in EPS_LIST:

            print(f"We are working for eps = {eps}")
            adv_image = image+eps*perturbations_per_epochs[i]

            result_perturbations_dir = VAL_PERTURBATIONS_DIR + f"/for_{eps}_{epochs}"
            save_image(perturbations, f"perturbations_{tail}", result_perturbations_dir)

            result_adversarial_dir = VAL_ADVERSARIAL_DIR + f"/for_{eps}_{epochs}"
            save_image(adv_image, f"adversarial_{tail}", result_adversarial_dir)
          i++
