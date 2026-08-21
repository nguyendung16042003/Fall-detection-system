#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
import cv2
from  modules import xconst
import numpy as np
from shapely.geometry import Polygon, Point

#print in message format
def xmsg(msg: str):
    print(f'[msg]: {msg}')

#print in error format
def xerr(error: str):
    print(f'[err]: {error}')

#check if value is not none
def isnn(value):
    return value is not None

#redraw the polygon
def add_polygon(frame, polygons, color=(255, 0, 0)):
    centroid = Polygon(polygons[0]).centroid
    for polygon_coords in polygons:
        pts = np.array(polygon_coords, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(frame, 'SAFE AREA' if xconst.IN_HOSPITAL else 'RISK AREA', (int(centroid.x), int(centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    return frame

#get width and height of text
def get_text_size(text, font_face, font_scale, thickness):
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    return text_width, text_height