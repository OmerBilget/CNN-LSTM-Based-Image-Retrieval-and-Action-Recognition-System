import collections
import glob
import os

import shutil
import sys
import random

CLASSNAMES=["Basketball","Biking","Bowling","HighJump","HorseRiding","SkyDiving","RopeClimbing","Skiing","TennisSwing","JavelinThrow","HammerThrow","GolfSwing"]
CLASSNAMES.sort()
TOTAL_LENGTH=100
VAL_SIZE= 20
TRAIN_SIZE= 70
TEST_SIZE= 10
DATASETNAME="Dataset2"
def get_class(fname):
    return fname.split('_')[-3]
def get_files_per_class(files):
    """Retrieve the files that belong to each class.

    Args:
        files: List of files in the dataset.

    Returns:
        Dictionary of class names (key) and files (values).
    """
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    #print(files_for_class)
    return files_for_class
def reset_directory(directoryName):
    relative_path = directoryName
    for item in os.listdir(relative_path):
        item_path = os.path.join(relative_path, item)

        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

def splitData(traincount,testcount,valcount,file,classname,datasetname):
    random.seed(777)
    random.shuffle(file)
    reset_directory(datasetname)
    for i in range(0,traincount+testcount+valcount):
        for j in classname:
            src= file[j][i]
            if i<traincount:
                dst = datasetname+"/train/"
            elif i<testcount+traincount:
                dst = datasetname+"/test/"
            elif i:
                dst = datasetname+"/val/"
            dst+=j
            if not os.path.exists(dst):
                os.makedirs(dst)
            dst=dst+"/"+src.split("/")[-1]
            print(src,"****",dst)
            shutil.copyfile(src, dst)



files=glob.glob("UCF-101/**/*.avi", recursive=True)
print(files)
files_for_class = get_files_per_class(files)
print(files_for_class)
splitData(TRAIN_SIZE,TEST_SIZE,VAL_SIZE,files_for_class,CLASSNAMES,DATASETNAME)
