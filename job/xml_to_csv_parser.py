import os
import csv
import glob
import ntpath
import shutil
import pandas as pd
import xml.etree.ElementTree as ET

ntpath.basename("a/b/c")

# Local Paths
# INPUT_XML_FILES_PATH = "C:\\Users\\Gopi Krishna\\Desktop\\live_projects\\MTech_Project\\XML_All_Annotations\\*.xml"
# FROM_IMAGE_FILES_PATH = "C:\\Users\\Gopi Krishna\\Desktop\\live_projects\\MTech_Project\\VOC_2012_All_JPG_Images"
# TO_IMAGE_FILE_PATH = "C:\\Users\\Gopi Krishna\\Desktop\\live_projects\\MTech_Project\\VOC_2012_Single_Label_JPG_Images"

# server paths
INPUT_XML_FILES_PATH = "/storage/home/gopikrishna/VOCdevkit/VOC2012/Annotations/*.xml"
FROM_IMAGE_FILES_PATH = "/storage/home/gopikrishna/VOCdevkit/VOC2012/JPEGImages"
TO_IMAGE_FILE_PATH = "/scratch/scratch2/gopi/SingleLabelImages"


column_names = [
    "Filename",
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
    "Total",
]


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def xmltocsv(xmlfile):
    dict = {
        "aeroplane": 0,
        "bicycle": 0,
        "bird": 0,
        "boat": 0,
        "bottle": 0,
        "bus": 0,
        "car": 0,
        "cat": 0,
        "chair": 0,
        "cow": 0,
        "diningtable": 0,
        "dog": 0,
        "horse": 0,
        "motorbike": 0,
        "person": 0,
        "pottedplant": 0,
        "sheep": 0,
        "sofa": 0,
        "train": 0,
        "tvmonitor": 0,
    }

    dict_mapping = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }

    arr = [
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

    tree = ET.parse(xmlfile)
    root = tree.getroot()
    srcpath = ""
    destpath = ""

    lines = []
    for i in range(21):
        lines.append(0)

    b = os.path.splitext(xmlfile)[0]
    b = path_leaf(b)

    temp = b + ".jpg"
    lines[0] = temp

    for name in root.iter("name"):
        i = 0
        a = name.text

        for i in range(0, 20):
            if a == arr[i] and dict[a] == 0:
                lines[i + 1] = 1
                dict[a] = dict[a] + 1

    with open("multi_label_csv_data.csv", "a") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(lines)


if __name__ == "__main__":
    print("***** Started Processing *****")

    # lines = []
    # for i in range(21):
    #     lines.append(0)

    lines = column_names[:21]

    with open("/scratch/scratch2/gopi/multi_label_csv_data.csv", "a") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(lines)

    input_files = glob.glob(INPUT_XML_FILES_PATH)

    for each_xml_file in input_files:
        xmltocsv(each_xml_file)

    print("Successfully processed input XML files.")

    multi_label_df = pd.read_csv("/scratch/scratch2/gopi/multi_label_csv_data.csv")
    multi_label_df["Total"] = multi_label_df.sum(axis=1)

    single_label_df = multi_label_df.loc[multi_label_df["Total"] == 1]

    single_label_df.reset_index(drop=True, inplace=True)
    # single_label_df.columns = column_names
    single_label_df.to_csv("/scratch/scratch2/gopi/single_label_csv_data.csv")
    single_label_jpg_file_name_list = single_label_df["Filename"].tolist()

    print("Successfully saved single label data to CSV file.")

    for each_file_name in single_label_jpg_file_name_list:
        source_file = FROM_IMAGE_FILES_PATH + f"/{each_file_name}"
        target_file = TO_IMAGE_FILE_PATH + f"/{each_file_name}"

        shutil.copy(source_file, target_file)

    print("Successfully copied single label image files to the new folder.")

    # I want a csv file where the 1st column is "image_filename" and the 2nd column is the label(0-19) corresponding to arr above

    single_label_df.drop("Total", inplace=True, axis=1)
    single_label_df.reset_index(drop=True, inplace=True)

    single_label_df = pd.read_csv("/scratch/scratch2/gopi/single_label_csv_data.csv")

    single_label_df["Label"] = single_label_df.apply(
        lambda row: row[row == 1].index[0]
        if row[row == 1].index[0] != "Unnamed: 0"
        else row[row == 1].index[1],
        axis=1,
    )

    new_df = single_label_df[["Filename", "Label"]]
    new_df.to_csv("/scratch/scratch2/gopi/labeled_data.csv")

    print("Successfully created a csv file with filename and label(0-19) ")
