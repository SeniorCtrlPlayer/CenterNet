from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multi_pose import MultiPoseDataset
from coco_hp import COCOHP


def get_dataset():
  class Dataset(COCOHP, MultiPoseDataset):
    pass
  return Dataset
  
