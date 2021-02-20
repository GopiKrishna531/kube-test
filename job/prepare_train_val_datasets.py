# We will first know how many SingleLabeledImages are there for each class and
# then we will create a dataset with train and val folders
import os
import glob
import shutil
import pandas as pd
from pprint import pprint


IMAGE_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

CLASSIFIED_IMAGES_SOURCE = "/scratch/scratch2/gopi/ClassifiedImages"
TRAIN_VAL_IMAGES_SOURCE = "/scratch/scratch2/gopi/TrainValSplitDataset"
TRAIN_IMAGES_SOURCE = "/scratch/scratch2/gopi/TrainValSplitDataset/train"
VAL_IMAGES_SOURCE = "/scratch/scratch2/gopi/TrainValSplitDataset/val"


PERCENT_IMAGE_VAL = 20


def prepare_train_val_dataset():

    single_label_df = pd.read_csv("/scratch/scratch2/gopi/single_label_csv_data.csv")
    result_dict = dict(zip(IMAGE_CLASSES, list(single_label_df.sum(axis=0))[-21:-1]))
    pprint(result_dict)

    for img_class, count in result_dict.items():
        image_move_count = round((PERCENT_IMAGE_VAL / 100) * count)

        os.mkdir(os.path.join(VAL_IMAGES_SOURCE, img_class))
        os.mkdir(os.path.join(TRAIN_IMAGES_SOURCE, img_class))

        ALL_IMAGES_FOLDER_PATH = CLASSIFIED_IMAGES_SOURCE + f"/{img_class}/*.jpg"

        all_image_files = glob.glob(ALL_IMAGES_FOLDER_PATH)

        val_images_list = all_image_files[:image_move_count]
        train_images_list = all_image_files[image_move_count:]

        for each_val_image in val_images_list:
            val_img_source = CLASSIFIED_IMAGES_SOURCE + f"/{img_class}/{each_val_image}"
            val_img_target = VAL_IMAGES_SOURCE + f"/{img_class}/{each_val_image}"
            shutil.copy(val_img_source, val_img_target)

        for each_train_image in train_images_list:
            train_img_source = (
                CLASSIFIED_IMAGES_SOURCE + f"/{img_class}/{each_train_image}"
            )
            train_img_target = TRAIN_IMAGES_SOURCE + f"/{img_class}/{each_train_image}"
            shutil.copy(train_img_source, train_img_target)


if __name__ == "__main__":
    print("Started preparing train, val dataset.")
    prepare_train_val_dataset()
    print("Finished preparing train, val dataset.")

# Total = {'aeroplane': 584,
#  'bicycle': 185,
#  'bird': 730,
#  'boat': 311,
#  'bottle': 185,
#  'bus': 154,
#  'car': 455,
#  'cat': 845,
#  'chair': 178,
#  'cow': 264,
#  'diningtable': 28,
#  'dog': 805,
#  'horse': 244,
#  'motorbike': 180,
#  'person': 5974,
#  'pottedplant': 145,
#  'sheep': 279,
#  'sofa': 98,
#  'train': 402,
#  'tvmonitor': 190}
