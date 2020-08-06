'''
dataset.py  
[age]_[gender]_[race]_[date&time].jpg
文件名称格式就是每张图像的标注信息

Age表示年龄，范围在0~116岁之间
Gender表示性别，0表示男性，1表示女性
Race表示人种

author:lh-13
date:20200802
'''

import torch 
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np 
import cv2 
import os 

class Dataset(Dataset):
    def __init__(self, root_dir):
        super(Dataset, self).__init__()
        #self.transform = transform.Compose([transforms.ToTensor()])    #????what
        #self.annotation_info = self.load_annotation(annotation_file)
        self.max_age = 116
        img_files = os.listdir(root_dir)
        nums = len(img_files)
        #age:0-116, 0:male 1:female
        self.ages = []
        self.genders = []
        self.images = []
        index = 0 
        for file_name in img_files:
            age_gender_group = file_name.split("_")
            age = age_gender_group[0]
            gender = age_gender_group[1]
            self.genders.append(np.float32(gender))
            self.ages.append(np.float32(age)/self.max_age)    #年龄规一化到0-1之间
            self.images.append(os.path.join(root_dir, file_name))
            index += 1 



    # def load_annotation(self, annotation_file):
    #     lines = []
    #     with open(annotation_file) as read_file:
    #         for line in read_file:
    #             line = line.replace('\n', '')
    #             lines.append(line)
    #     return lines


    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index):   #not need
            index = index.tolist()
            image_path = self.images[index]
        else:
            image_path = self.images[index]

        if not os.path.exists(image_path):
            raise KeyError("%s does not exist..." % image_path)
        img = cv2.imread(image_path)   #BGR order
        h, w, c = img.shape 
        #resize 
        img = cv2.resize(img, (64, 64))
        img = (np.float32(img)/ 255.0 - 0.5)/0.5  #-1,1之间 
        #h, w, c to c, h, w 
        img = img.transpose((2, 0, 1))
        sample = {'image':torch.from_numpy(img), 'age':self.ages[index], 'gender':self.genders[index]}
        
        return sample




        


    