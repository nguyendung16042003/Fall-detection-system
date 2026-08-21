#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
import sys

#change dir to import modules
sys.path.append('..')
from modules.xgeometic import extract_features
from modules.xenum import PoseType
from modules.xcustom_class import HistTracker

class BinaryClassifier(object):
    def __init__(self, bn_model):
        self.bn_model = bn_model
        self._graph = None
        self.extract_feat_names = ['center_mass_x', 
                                   'center_mass_y',
                                   'shoulder_nose_angle',
                                   'torso_angle', 
                                   'hip_angle',  
                                #    'width_height_ratio',
                                   'shoulder_angle',
                                   'left_leg_angle',
                                   'right_leg_angle',
                                   'nose_to_ankle_angle'
                                   ]
        self.hist_tracker = HistTracker(max_hist=200, g_shape=(1280, 720), num_feature=len(self.extract_feat_names))

    @property
    def graph(self):
        return self._graph

    def classify(self, keypoint):
        extracted_feats = extract_features(df=keypoint.df, 
                                           extract_feat_names=self.extract_feat_names).T

        # #normalize the df withoiut normalizer (not good idea)
        # df_min = extracted_feats.min()
        # df_max = extracted_feats.max()
        
        # # Apply the normalization formula
        # extracted_feats = 2 * ((extracted_feats - df_min) / (df_max - df_min)) - 1
        extracted_feats = extracted_feats.T

        self._graph = self.hist_tracker.push(extracted_feats)
        pred_pose = self.bn_model.predict(extracted_feats)
        if pred_pose == 1:
            return True, PoseType.STAND
        elif pred_pose == 0:
            return False, PoseType.LAYDOWN