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

data_test = MultiPose('/home/lwk/Documents/CenterNet/data')
# def get_dataset():
#   class Dataset(COCOHP, MultiPoseDataset):
#     pass
#   return Dataset
