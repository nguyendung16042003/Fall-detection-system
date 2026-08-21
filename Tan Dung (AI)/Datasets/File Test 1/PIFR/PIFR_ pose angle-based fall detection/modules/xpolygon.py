#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-29             #
#==============================#

#import all required modules
import cv2
import numpy as np
from copy import deepcopy
from .xenum import PolygonType
from .xutils import xmsg, xerr

class XPolygon(object):
    def __init__(self, cap, polygon_type=PolygonType.line, show_windows_size=(640, 480)):
        self.cap = cap
        self.drawing = False
        self.polygon_type = polygon_type
        self.show_windows_size = show_windows_size
        self.polygons = []
        self.points = []

        #get frame for drawing
        self.get_first_frame()

    def get_first_frame(self):
        xmsg('extract first frame for drawing polygon shape.')
        success, frame = self.cap.read()
        if not success: return
        self.frame = frame          

    def draw(self):
        xmsg('drawing risk area as polygon shape.')
        if self.polygon_type is PolygonType.line:
            return self.draw_line()
        elif self.polygon_type is PolygonType.curve:
            return self.draw_curve()

    def redraw_polygons(self):
        for polygon in self.polygons:
            cv2.polylines(img=self.frame, pts=[np.array(polygon[-2:])], isClosed=True, color=(0, 0, 255), thickness=2)
        for point in [i for j in self.polygons for i in j]:
            cv2.circle(img=self.frame, center=point, radius=5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

    def draw_line_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            t_point = deepcopy(self.points)
            if len(self.points) == 1: self.polygons.append(t_point)
            else:
                (p1, p2), (p3, p4) = (t_point[0], t_point[-1])
                diff = abs((p1 + p2) - (p3 + p4))
                self.polygons[len(self.polygons)-1] = t_point
                if diff < 3:
                    self.points = []
                    xmsg(f'polygon created. | points = {len(t_point) - 1}')
            self.redraw_polygons()

    def draw_line(self):
        self.frame = cv2.resize(self.frame, self.show_windows_size)
        ih, iw = self.frame.shape[:2]  #get frame height and width
        cv2.namedWindow('Draw Polygons')
        cv2.setMouseCallback('Draw Polygons', self.draw_line_callback)
        while True:
            cv2.imshow('Draw Polygons', self.frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        return self.polygons

    def redraw_curve(self):
        for polygon in self.polygons:
            for i in range(len(polygon) - 1):
                cv2.line(img=self.frame, pt1=polygon[i], pt2=polygon[i + 1], color=(0, 0, 255), thickness=2)

    def draw_curve_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                self.drawing = True
                self.polygons.append(self.points)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.points.append((x, y))
                self.redraw_curve()
 
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.points = []

    def draw_curve(self):
        self.frame = cv2.resize(self.frame, self.show_windows_size)
        ih, iw = self.frame.shape[:2]  #get frame height and width
        cv2.namedWindow('Draw Polygons')
        cv2.setMouseCallback('Draw Polygons', self.draw_curve_callback)
        while True:
            cv2.imshow('Draw Polygons', self.frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        return self.polygons