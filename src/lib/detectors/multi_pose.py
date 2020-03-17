from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

from models.model import create_model, load_model
from models.decode import multi_pose_decode
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

class MultiPoseDetector():
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.opt = opt
    self.pause = True
    self.flip_idx = opt.flip_idx

  def pre_process(self, image, meta=None):
    height, width = image.shape[0:2]
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([width / 2., height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      # pad is 127 or 31
      # while new_height < pad, inp_height is pad
      # while new_height > pad, inp_height is 2^n and n is the minist that 2^n > pad
      inp_height = (height | self.opt.pad) + 1
      inp_width = (width | self.opt.pad) + 1
      c = np.array([width // 2, height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    # resized_image = cv2.resize(image, (width, height))
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height),
                              flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    # writer = SummaryWriter()
    # writer.add_graph(self.model, images)
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      output['hm_hp'] = output['hm_hp'].sigmoid_()

      torch.cuda.synchronize()
      forward_time = time.time()
      
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        # reg=output['reg'], 
        hm_hp=output['hm_hp'],
        # hp_offset=output['hp_offset'],
        K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
      # import pdb; pdb.set_trace()
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    results[1] = results[1].tolist()
    return results

  def debug(self, debugger, images, dets, output):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
  
  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > self.opt.vis_thresh:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
    debugger.show_all_imgs(pause=self.pause)
    
  def run(self, image_or_path_or_tensor, meta=None):
    pre_time, net_time, dec_time, post_time = 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3))
    start_time = time.time()
    image = cv2.imread(image_or_path_or_tensor)
    detections = []
    pre_start_time = time.time()

    images, meta = self.pre_process(image, meta)
    images = images.to(self.opt.device)
    torch.cuda.synchronize()
    pre_process_time = time.time()
    pre_time += pre_process_time - pre_start_time
    
    output, dets, forward_time = self.process(images, return_time=True)

    torch.cuda.synchronize()
    net_time += forward_time - pre_process_time
    decode_time = time.time()
    dec_time += decode_time - forward_time
    
    if self.opt.debug >= 2:
      self.debug(debugger, images, dets, output)
    
    dets = self.post_process(dets, meta)
    torch.cuda.synchronize()
    post_process_time = time.time()
    post_time += post_process_time - decode_time

    detections.append(dets)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(debugger, image, results)
    
    return {'results': results, 'tot': tot_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}