import csv 
# import requests 
import xml.etree.ElementTree as ET 
# import cv2
import os

#import re
# from shutil import copyfile
from sys import exit
import glob
import errno

import pathlib
import shutil
import pandas as pd


import ntpath
ntpath.basename("a/b/c")

column_names = ["Filename", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair","cow", "diningtable", 
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa","train",
            "tvmonitor"] 


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def xmltocsv(xmlfile): 
    dict = {"aeroplane": 0, "bicycle" : 0, "bird" : 0, "boat": 0, "bottle" :0, 
            "bus":0, "car" : 0, "cat": 0 , "chair": 0 ,"cow": 0 , "diningtable": 0, 
            "dog": 0, "horse":0, "motorbike": 0, "person":0, "pottedplant":0, "sheep":0,"sofa":0,"train":0,
            "tvmonitor": 0} 
    
    arr = ["aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair","cow", "diningtable", 
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa","train",
            "tvmonitor"] 
    
    
    tree = ET.parse(xmlfile) 
    root = tree.getroot()
    srcpath=""
    destpath=""
    
    lines = []
    for i in range(21):
        lines.append(0)
    
    b=(os.path.splitext(xmlfile)[0])
    b=path_leaf(b)
    
    temp=b+".jpg"
    lines[0]=temp
    
    for name in root.iter('name'):
        i=0;
        a=name.text
        
        for i in range(0,20):
            if(a == arr[i] and dict[a]==0):
                lines[i+1]=1
                dict[a] =dict[a] + 1
                
    with open('multi_label_csv_data.csv', 'a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(lines)


# c=0
# #path = 'Annotations/*.xml'
# path = '/content/drive/MyDrive/sample_xml_files/xml_files_sample/*.xml'
# files = glob.glob(path)
# #print(files)
# for file in files:
#   #print("hai")
#   print(file)
#   xmltocsv(file)
#   c=c+1
#   print(c)


if __name__ == '__main__':
    print('***** Started Processing *****')

    lines = []
    for i in range(21):
        lines.append(0)

    with open('multi_label_csv_data.csv', 'a') as writeFile:
      writer = csv.writer(writeFile)
      writer.writerow(lines)

    # Local Paths
    # input_xml_files = "C:\\Users\\Gopi Krishna\\Desktop\\live_projects\\MTech_Project\\XML_All_Annotations\\*.xml"
    # from_image_file_path = "C:\\Users\\Gopi Krishna\\Desktop\\live_projects\\MTech_Project\\VOC_2012_All_JPG_Images"
    # to_image_file_path = "C:\\Users\\Gopi Krishna\\Desktop\\live_projects\\MTech_Project\\VOC_2012_Single_Label_JPG_Images"

    # server paths
    input_xml_files = "/storage/home/gopikrishna/VOCdevkit/VOC2012/Annotations/*.xml"
    from_image_file_path = "/storage/home/gopikrishna/VOCdevkit/VOC2012/JPEGImages"
    to_image_file_path = "/storage/home/gopikrishna/VOCdevkit/VOC2012/SingleLabelImages"


    input_files = glob.glob(input_xml_files)

    for each_xml_file in input_files:
        xmltocsv(each_xml_file)

    print('Successfully processed XML files.')

    multi_label_df = pd.read_csv('multi_label_csv_data.csv')
    multi_label_df['Total'] = multi_label_df.sum(axis=1)

    single_label_df = multi_label_df.loc[multi_label_df['Total'] == 1]

    single_label_df.reset_index(drop=True, inplace=True)
    single_label_df.to_csv('single_label_csv_data.csv')

    single_label_jpg_file_name_list = single_label_df['0'].tolist()

    print('Successfully saved single label data to CSV file.')

    for each_file_name in single_label_jpg_file_name_list:
        source_file = from_image_file_path + f"\\{each_file_name}"
        target_file = to_image_file_path + f"\\{each_file_name}"

        shutil.copy(source_file, target_file)

    print('Successfully copied single label image files to the new folder.')

