#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
import sys
from shapely.geometry import Polygon, Point
from .xbinary_classifer import BinaryClassifier

#change dir to import modules
sys.path.append('..')
from modules.xutils import xmsg
from modules.xcustom_class import History
from modules.person_pose import PersonInfo
from modules import xconst
from modules.xenum import PoseType, FallConfig
# from modules.xgeometic import extract_features

class FallDetectionYOLO(object):
    def __init__(self,
                 risk_areas,
                 falling_criteria=FallConfig.TIME_THRESHOLD
                 ):
        self.risk_areas = Polygon(risk_areas[0])
        self.falling_criteria = falling_criteria
        self.history = History(max_track=200)
        xmsg(f'polygon: {self.risk_areas}')

    def prepare_data(self, result):
        boxes, track_ids, plot = [], [], None
        result = result.result
        if result:
            boxes = result.boxes.xywh.cpu()
            if result.boxes.id != None:
                track_ids = result[0].boxes.id.int().cpu().tolist()
            plot = result.plot(**xconst.TRACK_PARAMS)
        return boxes, track_ids, plot

    #determine a fall based on criteria
    def determine_fall(self, track_id):
        lying_percentage = self.history.data[track_id].lying_percent
        stable_detect = self.history.data[track_id].stable_detect
        if self.falling_criteria is FallConfig.FRAME_THRESHOLD:
            return (lying_percentage > 50) and stable_detect

    def detect(self, result):
        self.result = result
        boxes, track_ids, plot = self.prepare_data(self.result)
        for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
            xc, yc, w, h = list(map(lambda x: int(x), box))
            if xconst.IN_HOSPITAL:
                if Point((xc, yc)).within(self.risk_areas):
                    continue #skip if there is fall in the safe area
            self.history.push_pose(track_id=track_id, bn_pose=PoseType.LAYDOWN) #all detected object are always lay down
            self.history.push_loc(track_id=track_id, loc=(xc, yc, w, h))
            isfall = self.determine_fall(track_id)
            if isfall: result.data.fall[idx] = 1
        result.data['history'] = self.history
        return self.result

#=================================================================================================#
#================================= For Feature Extraction Method =================================#
#=================================================================================================#

#base class for danger detection
class FallDetection(object):
    def __init__(self, 
                 risk_areas, #polygon coordinate represent risk area
                 bn_model, #binary model for classifying standing/lying
                 falling_criteria=FallConfig.TIME_THRESHOLD): #criteria for determining fall
        
        self.risk_areas = Polygon(risk_areas[0])
        self.bn_model = BinaryClassifier(bn_model=bn_model)
        self.falling_criteria = falling_criteria
        self.history = History(max_track=100)
        xmsg(f'polygon: {self.risk_areas}')

    #process yolo result formated format
    def prepare_yolo_result(self, result=None):
        boxes, track_ids, keypoints_list, plot = [], [], [], None
        result = result.result
        if result:
            boxes = result.boxes.xywh.cpu()
            if result.boxes.id != None:
                track_ids = result[0].boxes.id.int().cpu().tolist()
            
            plot = result.plot(**xconst.TRACK_PARAMS)
            for keypoints in result.keypoints:
                keypoints = PersonInfo(keypoints.xyn.cpu().numpy()[0])
                keypoints_list.append(keypoints)
        return boxes, track_ids, keypoints_list, plot
    
    #determine a fall based on criteria
    def determine_fall(self, track_id):
        lying_percentage = self.history.data[track_id].lying_percent
        if self.falling_criteria is FallConfig.FRAME_THRESHOLD:
            return lying_percentage > 50

    def detect(self, result):
        self.result = result
        boxes, track_ids, keypoints, plot = self.prepare_yolo_result(self.result)
        for idx, (box, track_id, keypoint) in enumerate(zip(boxes, track_ids, keypoints)):
            xc, yc, w, h = list(map(lambda x: int(x), box))
            # keypoint.id, keypoint.XYWH = track_id, [xc, yc, w, h]
            keypoint.add_meta(xywh=(xc, yc, w, h))
            error, decision = self.bn_model.classify(keypoint)
            self.history.push(track_id=track_id, bn_pose=decision)
            result.data['history'] = self.history

            isfall = self.determine_fall(track_id)
            if isfall: result.data.fall[idx] = 1
        return self.result