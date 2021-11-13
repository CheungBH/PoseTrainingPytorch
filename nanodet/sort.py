"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import argparse
import warnings

warnings.filterwarnings('ignore')
from filterpy.kalman import KalmanFilter


def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return o


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))


def obtain_id(c_ids):
  begin = 0
  while True:
    begin += 1
    if begin not in c_ids:
      return begin


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  curr_id = []
  # print("Count is {}".format(count))
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01
    self.kf.x[:4] = convert_bbox_to_z(bbox)

    self.time_since_update = 0
    self.id = obtain_id(KalmanBoxTracker.curr_id)
    KalmanBoxTracker.curr_id.append(self.id)
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.objclass = bbox[6]

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

# Sort begins


class Sort(object):
  def __init__(self, max_age=1, min_hits=0):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.iou_matrix = []
    self.id2pred = {}
    self.mat = [[]]
    self.match_indices = []
    self.tracker_pos = {}

  def init_KF(self):
    KalmanBoxTracker.count = 0
    KalmanBoxTracker.curr_id = []
    self.trackers = []

  def draw_iou_mat(self, interval=8):
    if len(self.iou_matrix) < 1:
      return [["N/A"]]
    iou = self.iou_matrix.T
    matrix = [["t\d".rjust(interval, " ")]]
    dets_ls = [str(idx).rjust(interval, " ") for idx in range(len(iou[0]))]
    matrix[0] += dets_ls
    for idx, item in enumerate(iou):
      trks_ls = [str(self.tracker_pos[idx]).rjust(interval, " ")]
      trks_ls += [str(round(tmp, 2)).rjust(interval, " ") for tmp in item]
      matrix.append(trks_ls)
    # mat_str = [ls2str(item) for item in matrix]
    return matrix

  def associate_detections_to_trackers(self,detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
      return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    self.iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
      for t, trk in enumerate(trackers):
        self.iou_matrix[d, t] = iou(det, trk)
    self.match_indices = linear_assignment(-self.iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
      if (d not in self.match_indices[:, 0]):
        unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
      if (t not in self.match_indices[:, 1]):
        unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in self.match_indices:
      if (self.iou_matrix[m[0], m[1]] < iou_threshold):
        unmatched_detections.append(m[0])
        unmatched_trackers.append(m[1])
      else:
        matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
      matches = np.empty((0, 2), dtype=int)
    else:
      matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

  def update(self, dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    self.id2pred = {}
    self.tracker_pos = {}
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
        pos = self.trackers[t].predict()[0]
        idx = self.trackers[t].id
        self.id2pred[idx] = pos
        self.tracker_pos[t] = idx
        trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        if np.any(np.isnan(pos)):
            to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
        self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
    self.mat = self.draw_iou_mat()

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
        if t not in unmatched_trks:
            d = matched[np.where(matched[:,1]==t)[0],0]
            trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            ret.append(np.concatenate((d,[trk.id], [trk.objclass])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if trk.time_since_update > self.max_age or \
                (trk.age < self.min_hits * 0.5 and trk.time_since_update > 0.3 * self.max_age):
            KalmanBoxTracker.curr_id.remove(self.trackers[i].id)
            self.trackers.pop(i)

    # KalmanBoxTracker.curr_id = [trks.id for trks in self.trackers]

    if len(ret) > 0:
        return np.concatenate(ret)
    return np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  tracked_boxes = [[100,100,200,200,1,0,0], [400,300,410,320,1,0,0],[150,170,140,170,1,0,0]]
  dets_boxes = [[90,90,180,180,1,0,0], [110,110,190,190,1,0,0]]
  associate_detections_to_trackers(tracked_boxes, dets_boxes)


