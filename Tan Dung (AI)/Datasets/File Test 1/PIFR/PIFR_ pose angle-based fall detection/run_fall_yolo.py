#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-06-10             #
#==============================#

#import all required modules
from glob import glob
import random
from detection.core import DLModelYOLO
from modules.xcustom_class import Config
from modules.xstream import Stream
from modules.xpolygon import XPolygon
from modules.xutils import xmsg, xerr
from modules.xenum import StreamType, ModelType, PolygonType

#strating point of the detection
if __name__ == '__main__':
    #global vars
    config = Config()
    config.cam_windows_size = (1280, 720)
    config.show_windows_size = (1280, 720)

    #define stream object
    stream = Stream(stream_type = StreamType.file, config = config)

    #pick random video from ./assets/videos/*
    video_list = glob('./assets/videos/*/*', recursive=True)
    pick_video = random.choice(video_list)
    xmsg(f'pick video: {pick_video}')
    stream.set_file_location(pick_video)
    cap = stream.get_cap()

    #configure risk are polygon
    polygon = XPolygon(cap=cap, 
                       polygon_type=PolygonType.line, 
                       show_windows_size=config.show_windows_size)
    polygons_list = polygon.draw()
    # polygons_list=[[(446, 345), (867, 364), (903, 514), (400, 493), (446, 345)]]

    #start prediction
    dlmodel = DLModelYOLO(model_type=ModelType.yolov8n,
                      stream=stream,
                      config=config)
    
    dlmodel.set_risk_area(polygons_list)
    dlmodel.detect(extract=True, save_file=True)