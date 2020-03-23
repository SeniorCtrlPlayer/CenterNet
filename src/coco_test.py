# -*- coding: utf-8 -*-

import pycocotools.coco as coco
import _init_paths
from torch.utils.data import DataLoader
from lib.datasets.multi_pose import MultiPoseDataset
from lib.datasets.coco_hp import COCOHP
from opts import opts
import os

class MultiPose(MultiPoseDataset):
    def __init__(self, data_dir):
        super().__init__()
        split = 'train'
        self.data_dir = os.path.join(data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        if split == 'test':
          self.annot_path = os.path.join(
              self.data_dir, 'annotations', 
              'image_info_test-dev2017.json').format(split)
        else:
          self.annot_path = os.path.join(
            self.data_dir, 'annotations', 
            'person_keypoints_{}2017.json').format(split)
        self.coco = coco.COCO(self.annot_path)
        self.image_ids = self.coco.getImgIds()
        img_id = self.image_ids[0]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

def verify(data_dir, split):
    split = 'train'
    data_dir = os.path.join(data_dir, 'coco')
    img_dir = os.path.join(data_dir, '{}2017'.format(split))
    if split == 'test':
      annot_path = os.path.join(
          data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split)
    else:
      annot_path = os.path.join(
        data_dir, 'annotations', 
        'person_keypoints_{}2017.json').format(split)
    coco_test = coco.COCO(annot_path)
    image_ids = coco_test.getImgIds()
    flag = False
    no_exists_filename = []
    small_filename = []
    for img_id in image_ids:
        file_name = coco_test.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(img_dir, file_name)
        try:
            img_size = os.path.getsize(img_path)/float(1024)
            if img_size < 5:
                small_filename.append(img_path)
        except OSError:
            flag = True
            no_exists_filename.append(img_path)
            break
    if flag:
        print(no_exists_filename)
        print(small_filename)
    else:
        print(small_filename)
        print(split+' is complete')

# data_test = MultiPose('/home/lwk/Documents/CenterNet/data')
verify('/home/lwk/Documents/CenterNet/data', 'train')
