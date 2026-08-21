#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-05-07             #
#==============================#

#import all required modules
import numpy as np
import pandas as pd
from .xenum import PoseType
import cv2, math, matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TKAgg')

#class for max size list
class MSL:
    def __init__(self, max_size):
        self.max_size = max_size
        self.elements = []

    def __len__(self):
        return len(self.elements)

    def is_full(self):
        return len(self.elements) == self.max_size

    def push(self, element):
        if self.is_full():
            self.elements.pop(0)  # Remove the front element if full
        self.elements.append(element)

    def push_all(self, elements):
        for element in elements:
            self.push(element)

    def empty(self):
        self.elements = []

    def get_list(self):
        return self.elements.copy()
    
#store configuration/variable for inner/outer function use
class Config():
    def __init__(self):
        #dictionary to store dynamic variables
        self._dynamic_variables = {}

    def __getattr__(self, name):
        #this method is called when an attribute is not found
        if name in self._dynamic_variables:
            return self._dynamic_variables[name]
        else:
            return None
            
    def __setattr__(self, name, value):
        #this method is called when an attribute is set
        if name == '_dynamic_variables':
            #allow setting the _dynamic_variables attribute directly
            super().__setattr__(name, value)
        else:
            #set a dynamic variable
            self._dynamic_variables[name] = value

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        self.set(name, value)

    def set(self, name, value):
        self._dynamic_variables[name] = value

    def contain_key(self, name):
        return name in self._dynamic_variables

class HistTracker:
    def __init__(self, max_hist=60, g_shape=(720, 1280), num_feature=9):
        self.feature_names = [
            "center_mass_x", "center_mass_y", "shoulder_nose_angle", 
            "torso_angle", "hip_angle", "shoulder_angle", 
            "left_leg_angle", "right_leg_angle", "nose_to_ankle_angle"
        ]
        self.num_feature = num_feature
        self.g_shape = g_shape
        self.values = MSL(max_size=max_hist)
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.first_inference = True

        # Create a line plot for each geometric feature
        self.lines = None

    def push(self, new_val):
        if self.first_inference:
            self.lines = [self.ax.plot([], [], marker='', label=new_val.columns.to_list()[i])[0] for i in range(self.num_feature)]
            self.ax.set_xlabel('Time (seconds)')
            self.ax.set_ylabel('Feature Value')
            self.ax.set_title('Geometric feature Changes Over Time')
            self.ax.legend(loc='upper left')
            # self.ax.grid(True)
            self.fig.tight_layout()
            self.first_inference=False
        # Validate the input DataFrame
        if not isinstance(new_val, pd.DataFrame):
            raise ValueError("new_val must be a pandas DataFrame")
        if new_val.shape != (1, self.num_feature):
            print(new_val.shape)
            raise ValueError(f"new_val must have exactly one row and {self.num_feature} columns")

        # Convert the DataFrame to a numpy array
        new_val_np = new_val.to_numpy().reshape(1, -1)
        
        # Push the new value to the history
        self.values.push(new_val_np.tolist()[0])
        
        # Convert the history to a numpy array
        features_np = np.asarray(self.values.get_list())
        time = np.arange(len(self.values))

        # Update each line plot with new data
        for feature, line in enumerate(self.lines):
            line.set_data(time, features_np[:, feature])

        # Adjust plot limits
        self.ax.set_xlim(0, max(200, len(self.values)))
        self.ax.set_ylim(np.min(features_np), np.max(features_np))
        self.ax.set_ylim(0, np.max(features_np)+10)

        # Convert the figure to a numpy array
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (self.fig.canvas.get_width_height()[::-1] + (3,))

        # Resize the image
        graph = cv2.resize(buf, self.g_shape, interpolation=cv2.INTER_LINEAR)
        return graph

    def gen_empty(self):
        empty_values = np.zeros((self.values.max_size, self.num_feature)).tolist()
        self.values.push_all(empty_values)

#base class for storing history across frames
class History(object):
    def __init__(self, 
                 max_track=100 #number of frames to keep track
                 ):
        self.max_track = max_track
        self.data = Config()

    #create or add new element to history track
    def push_pose(self, track_id, bn_pose):
        if not self.data.contain_key(track_id):
            self.data[track_id] = self.DetailHistory(self.max_track)
        self.data[track_id].pose.push(bn_pose)
        self.data[track_id].cal_lying_percent()
    
    #create or add new element to history track
    def push_loc(self, track_id, loc):
        if not self.data.contain_key(track_id):
            self.data[track_id] = self.DetailHistory(self.max_track)
        self.data[track_id].loc.push(loc)
        self.data[track_id].cal_stable_detect()
    
    def clear(self):
        self.data = Config()

    #keep track of detail pose for each tracked person object
    class DetailHistory(object):
        def __init__(self, max_track):
            self.pose = MSL(max_size=max_track)
            self.loc = MSL(max_size=100)
            self.lying_percent = 0
            self.stable_detect = False

        #calculate the percentage of lying in the last nth frames
        def cal_lying_percent(self):
            lying_count = [i for i in self.pose.elements if i == PoseType.LAYDOWN]
            self.lying_percent = round(len(lying_count) * 100 / len(self.pose.elements), 2)

        def cal_stable_detect(self):
            coords = [(xc, yc) for (xc, yc, w, h) in self.loc.get_list()]
            _,_,w, _ = self.loc.get_list()[-1]
            if len(coords) < 20: #need at least 20 frames before checking for stable dection to save computational cost
                return False

            first_point = coords[0]
            last_point = coords[-1]
            actual_distance = self.euclidean_distance(first_point, last_point)
            self.stable_detect = actual_distance < w

        def euclidean_distance(self, p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def bounding_box_max_distance(self, coords):
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]
            max_dist = self.euclidean_distance((min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)))
            return max_dist