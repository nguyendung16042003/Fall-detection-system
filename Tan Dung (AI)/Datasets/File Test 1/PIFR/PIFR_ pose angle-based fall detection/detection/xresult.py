#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-05-07             #
#==============================#

#import all required modules
import sys
import numpy as np

#change dir to import modules
sys.path.append('..')
from shapely.geometry import Point, Polygon
from modules.xcustom_class import Config

class XResult(object):
    def __init__(self, result):
        self.result = result
        self._xboxes = []
        self.data = Config()
        self.data['fall'] = [0] * len(result.boxes.cls)
    
    @property
    def xboxes(self):
        if len(self._xboxes) == 0: 
            self._xboxes = [XBoxes(boxes) for boxes in self.result.boxes]
        return self._xboxes
    
    @xboxes.setter
    def xboxes(self, d):
        self._xboxes = d

    @property
    def boxes(self):
        return self.result.boxes
        
    @property
    def img(self):
        return self.result.orig_img

    @property
    def cls(self):
        return self.boxes.cls.cpu().numpy()

    @property
    def conf(self):
        return self.boxes.conf

    @property
    def ids(self):
        return self.boxes.id.cpu().numpy().astype(int) if self.boxes.id is not None else None

    @property
    def names(self):
        return self.result.names

    @property
    def xyxy(self):
        return self.boxes.xyxy.cpu().numpy().astype(int)
    
    @property
    def xywh(self):
        return self.boxes.xywh.cpu().numpy().astype(int)
    
    @property
    def bottom_xcyc(self):
        xcyc = np.empty((len(self.boxes), 2), dtype=int)
        for i, row in enumerate(self.xyxy):
          x1, y1, x2, y2 = row
          xc = int((x1 + x2) / 2)
          yc = (y2 - 5) #5pixel from the bottom of the bounding box
          xcyc[i, 0] = xc
          xcyc[i, 1] = yc
        return xcyc
    
    @property
    def xcyc(self):
        xcyc = np.empty((len(self.boxes), 2), dtype=int)
        for i, row in enumerate(self.xyxy):
          x1, y1, x2, y2 = row
          xc = int((x1 + x2) / 2)
          yc = int((y1 + y2) / 2)
          xcyc[i, 0] = xc
          xcyc[i, 1] = yc
        return xcyc
    
    def filter_out_polygon(self, polygon):
        filtered_indices = []
        for i, (xc, yc) in enumerate(self.xcyc):
            point = Point(xc, yc)
            if not polygon.contains(point):
                filtered_indices.append(i)

        # Create a new filtered result
        filtered_result = self._filter_by_indices(filtered_indices)
        return XResult(filtered_result)

    def _filter_by_indices(self, indices):
        # Create a new result object with filtered data
        new_result = self.result.copy()
        new_result.boxes = self.result.boxes[indices]
        return new_result
    
class XBoxes(object):
    def __init__(self, boxes):
        self.boxes = boxes
        self.is_danger = False