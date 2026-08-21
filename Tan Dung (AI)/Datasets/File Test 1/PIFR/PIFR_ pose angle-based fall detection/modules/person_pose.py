#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-06-25             #
#==============================#

#import all required modules
import pandas as pd
from .xenum import KEYPOINT_ENUM
from .xcustom_class import Config

class PersonInfo(object):
    def __init__(self, pose_info):
        self.pose = Config()
        self.meta = Config()
        self._df = None
        self.add_pose_info(pose_info=pose_info)
        
    @property
    def df(self):
        self.prepare_df()
        return self._df
    
    def add_meta(self, xywh):
        self.meta['xywh'] = xywh

    def add_pose_info(self, pose_info):
        keypoints_names = [name for name in vars(KEYPOINT_ENUM) if not name.startswith("__")]
        for keypoint_name in keypoints_names:
            try:
                setattr(self, f'{keypoint_name.lower()}_x', pose_info[getattr(KEYPOINT_ENUM, keypoint_name)][0])
                setattr(self, f'{keypoint_name.lower()}_y', pose_info[getattr(KEYPOINT_ENUM, keypoint_name)][1])
            except:
                pass
    
    def prepare_df(self):
        keypoints_names = [name for name in vars(KEYPOINT_ENUM) if not name.startswith("__")]
        #set pose data into df
        result_dict = {}
        for keypoint_name in keypoints_names:
            try:
                result_dict[f'{keypoint_name.lower()}_x'] = getattr(self, f'{keypoint_name.lower()}_x')
                result_dict[f'{keypoint_name.lower()}_y'] = getattr(self, f'{keypoint_name.lower()}_y')
            except:
                pass
        result_dict['xywh'] = self.meta.xywh
        df = pd.DataFrame(result_dict.items()).T
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        self._df = df if (df.shape[1] > 2) else None

# class PERSON_POSE():
#     def __init__(self, pose_info):
#         keypoints_names = [name for name in vars(KEYPOINT_ENUM) if not name.startswith("__")]
#         for keypoint_name in keypoints_names:
#             try:
#                 setattr(self, f'{keypoint_name.lower()}_x', pose_info[getattr(KEYPOINT_ENUM, keypoint_name)][0])
#                 setattr(self, f'{keypoint_name.lower()}_y', pose_info[getattr(KEYPOINT_ENUM, keypoint_name)][1])
#             except:
#                 pass

#     def get_df(self, fn, pose='stand'):
#         keypoints_names = [name for name in vars(KEYPOINT_ENUM) if not name.startswith("__")]
#         df_keypoints_names = ['fn'] +[i for i in keypoints_names if 'xywh' not in i]+ ['pose']
#         result_dict = {}
#         for keypoint_name in keypoints_names:
#             try:
#                 result_dict['fn'] = fn
#                 result_dict[f'{keypoint_name.lower()}_x'] = getattr(self, f'{keypoint_name.lower()}_x')
#                 result_dict[f'{keypoint_name.lower()}_y'] = getattr(self, f'{keypoint_name.lower()}_y')
#                 result_dict['pose'] = pose
#             except:
#                 pass
#         df = pd.DataFrame(result_dict.items()).T
#         new_header = df.iloc[0]
#         df = df[1:]
#         df.columns = new_header
#         return df if (df.shape[1] > 2) else None 

#     def get_df_pred(self):
#         keypoints_names = [name for name in vars(KEYPOINT_ENUM) if not name.startswith("__")]
#         df_keypoints_names = [i for i in keypoints_names if 'xywh' not in i]
#         result_dict = {}
#         for keypoint_name in keypoints_names:
#             try:
#                 result_dict[f'{keypoint_name.lower()}_x'] = getattr(self, f'{keypoint_name.lower()}_x')
#                 result_dict[f'{keypoint_name.lower()}_y'] = getattr(self, f'{keypoint_name.lower()}_y')
#             except:
#                 pass
#         df = pd.DataFrame(result_dict.items()).T
#         new_header = df.iloc[0]
#         df = df[1:]
#         df.columns = new_header
#         return df if (df.shape[1] > 2) else None 