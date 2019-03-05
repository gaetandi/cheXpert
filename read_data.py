# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
import csv
from torch.utils.data import Dataset
from PIL import Image



class CheXpertDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            for line in csvReader:
                image_name= line[0]
                label = line[5:]
                for l in label:
                    if l:
                        l = int(float(l))
                    else:
                        l = 0
                #image_name = os.path.join(data_dir, image_name)
                image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

