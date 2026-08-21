#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
import cv2
from .xcustom_class import Config
from .xenum import BBType

################# directory constants #################
DL_MODEL_ROOT='./assets/models'
DL_CUSTOM_MODEL_ROOT='./assets/models/custom/v1'
BN_MODEL_ROOT='./assets/models/binary' #binary classifier models path
PRED_SAVE_DIR='./assets/predictions'

################# deep learning model #################
#yolo classes for detection
DETECT_YOLO_CLASS=[0] #fall class
DETECT_YOLO_CONF=0.3 #min confident for yolo model
TRACK_PARAMS = {
    'font_size': 20,
    'kpt_radius':0,
    'kpt_line': False,
    'labels': False,
    'line_width': 1,
    }

################# color constants #################
#constant value for plotting
color = Config()
color.GREENISH=(0, 181, 6)
color.RED=(0, 0, 255)
color.WHITE=(255, 255, 255)

################# plotting configuratiion #################
plot_config = Config()
plot_config.bb=True
plot_config.cls_name=False
# plot_config.bottom_point=True
plot_config.lying_percent=False
plot_config.fall_alert=True
plot_config.bottom_ellipse=True
plot_config.bb_type=BBType.CORNER #CORNER, ELLIPSE, NORMAL
plot_config.only_in_risk_area=False #plot the bounding box for any object detected in risk area

#plotting color
plot_config.color=Config()
plot_config.color.bb=color.GREENISH
plot_config.color.cls_name=color.RED
plot_config.color.fall_alert_bg=color.RED
plot_config.color.fall_alert_text=color.WHITE
plot_config.color.risk_area_border=color.RED

#thickness
plot_config.thickness=Config()
plot_config.thickness.bb=2
plot_config.thickness.cls_name=2
plot_config.thickness.lying_percent=1
plot_config.thickness.fall_alert_text=2

#font family
plot_config.font=Config()
plot_config.font.all=cv2.FONT_HERSHEY_SIMPLEX

#font scale
plot_config.font_scale=Config()
plot_config.font_scale.cls_name=0.8
plot_config.font_scale.lying_percent=0.6
plot_config.font_scale.fall_alert_text=0.9

################# Utils #################
ADD_GRAPH=True
DEVELOPMENT=False
IN_HOSPITAL=False