import pandas as pd
import shutil
import os
import traceback

# server paths
FROM_IMAGE_FILES_PATH = "/scratch/scratch2/gopi/SingleLabelImages"
TO_IMAGE_FILE_PATH = "/scratch/scratch2/gopi/ClassifiedImages"

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


def process_and_generate_images_per_class():
    try:
        label_data_df = pd.read_csv("/scratch/scratch2/gopi/labeled_data.csv")

        print("Creating sub folders with the image classification names.")

        for each_class_folder in IMAGE_CLASSES:
            os.mkdir(os.path.join(TO_IMAGE_FILE_PATH, each_class_folder))

        print("Copying Images to the corresponding Image classification folder..")

        for idx, row in label_data_df.iterrows():
            source_file = FROM_IMAGE_FILES_PATH + f"/{row['Filename']}"
            target_file = (
                TO_IMAGE_FILE_PATH + f"/{row['Label']}" + f"/{row['Filename']}"
            )

            shutil.copy(source_file, target_file)
    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    print(" Started processing..!!")
    process_and_generate_images_per_class()
    print(" Finished processing..!!")
