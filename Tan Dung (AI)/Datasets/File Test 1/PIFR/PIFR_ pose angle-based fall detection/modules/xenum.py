#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
from enum import Enum

#enum for type of stream
class StreamType(Enum):
    usb = 'usb'
    csi = 'csi'
    file = 'file'

    def __str__(self):
        return self.value

#enum for yolo model type
class ModelType(Enum):
    yolov8n = 'yolov8n'
    yolov8s = 'yolov8s'
    yolov8m = 'yolov8m'
    yolov8l = 'yolov8l'
    yolov8x = 'yolov8x'
    yolov8n_pose = 'yolov8n-pose'
    yolov8s_pose = 'yolov8s-pose'
    yolov8m_pose = 'yolov8m-pose'
    yolov8l_pose = 'yolov8l-pose'
    custom_best = 'best'

    def __str__(self):
        return self.value
    
class BNModelType(Enum):
    svm = 'svm_model'

    def __str__(self):
        return self.value

#enum for polygon type
class PolygonType(Enum):
    line = 'line'
    curve = 'curve'

    def __str__(self):
        return self.value
    
class KEYPOINT_ENUM():
    # ID:             int = -1
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16
    XYWH:           int = 17

class SLMethod(Enum):
    WHRatio = 0
    LAYDOWN = 1

class PoseType(Enum):
    STAND = 0
    LAYDOWN = 1

class FallConfig(Enum):
    FRAME_THRESHOLD = 0
    TIME_THRESHOLD = 1
    SPEED_THRESHOLD = 2

class BBType(Enum):
    NORMAL = 0
    CORNER = 1
    ELLIPSE = 2