#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
import numpy as np
from .xresult import XResult
from ultralytics import YOLO
from datetime import datetime
from shapely.geometry import Point, Polygon
import sys, cv2, os, uuid, time, joblib, torch, gc
from .fall_detection import FallDetection, FallDetectionYOLO

#change dir to import modules
sys.path.append('..')
from modules import xconst
from modules.xplot import XPlot
from modules.xenum import ModelType, StreamType, BNModelType, FallConfig
from modules.xutils import xmsg, xerr, add_polygon

class DLModelYOLO(object):
    def __init__(self, 
                 model_type=ModelType.yolov8n,
                 stream=None,
                 config=None):
        self.model_type = model_type
        self.stream = stream
        self.config = config
        self.model = None
        self.preds = []

    def load_yolo_model(self):
        xmsg('loading yolo model from .pt file.')
        self.model = YOLO(f'{xconst.DL_CUSTOM_MODEL_ROOT}/{self.model_type.value}.pt')
        xmsg('yolo model loaded successfully.')
        return self.model
    
    #add polygon to the prediction
    def set_risk_area(self, coords):
        self.risk_areas = coords

    def set_dangerD(self):
        self.fallD = FallDetectionYOLO(risk_areas=self.risk_areas,
                                   falling_criteria=FallConfig.FRAME_THRESHOLD
                                   )
        
    #save the prediction result into mp4 file
    def save_prediction(self, filename):
        #validate the stream type
        if self.stream.stream_type is not StreamType.file:
            xerr('only stream_type=file can be saved as a file.')
            return 0
        xmsg(f'wait - start saving prediction into file: "{filename}"')

        #prepare videowriter object
        _, height, width, _ = np.array(self.preds[:1]).shape
        codec_id = "mp4v" # ID for a video codec.
        fourcc = cv2.VideoWriter_fourcc(*codec_id)
        out = cv2.VideoWriter(os.path.join(xconst.PRED_SAVE_DIR, filename), fourcc=fourcc, fps=30, frameSize=(width, height))

        #write frames into file one by one
        xmsg('start writing frames into mp4 file.')
        start_time = time.time()
        for pred_frame in self.preds:
            out.write(pred_frame)
        out.release()

        #calculate saving time
        end_time = time.time()  # Record end time
        saving_time = end_time - start_time
        xmsg(f'file successfully saved as: "{filename}" | Saving time: {saving_time} seconds')

    #predict the list of frames extracted from video file
    def detect(self, extract=True, save_file = False, filename=None):

        #load models if it is not loaded yet
        if not self.model: self.load_yolo_model()
        #extract the frames before prediction instread of realtime prediction
        if (self.stream.stream_type is StreamType.file) and extract:
            frames = self.stream.extract_frames()
            self.set_dangerD()
            #iterate over all frames
            for idx, frame in enumerate(frames):
                result = self.model.track(frame, 
                                    verbose=False, 
                                    classes=xconst.DETECT_YOLO_CLASS, 
                                    conf=xconst.DETECT_YOLO_CONF)[0]
                pred_plot = result.plot()
                if len(result) > 0:
                    xresult = XResult(result=result)
                    # outside_polygon_res = xresult.filter_out_polygon(Polygon(self.risk_areas[0]))
                    # print(len(outside_polygon_res))
                    res = self.fallD.detect(result=xresult)
                    pred_plot = XPlot(result=res, config=xconst.plot_config, risk_areas=self.risk_areas).plot()

                if self.risk_areas: pred_plot = add_polygon(pred_plot, self.risk_areas, color=xconst.plot_config.color.risk_area_border)
                self.preds.append(pred_plot)
                
                cv2.imshow('Prediction - Frame extracted', pred_plot)
                # time.sleep(0.033)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        else:
            pass #for different kinds of stream (usb camera,...)
        
        #clean the cv2 and torch after prediction completed.
        self.model.cpu()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        cv2.destroyAllWindows()

        #save prediction into mp4 file
        if save_file:
            if not filename:
                unique_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{unique_id[:5]}_{timestamp}.mp4"
            self.save_prediction(filename)

#=================================================================================================#
#================================= For Feature Extraction Method =================================#
#=================================================================================================#

#class handler for deep learning model operation
class DLModel():
    def __init__(self, 
                 model_type = ModelType.yolov8n, 
                 bn_model_type=BNModelType.svm, 
                 stream = None, 
                 config=None):
        
        self.model_type = model_type
        self.bn_model_type = bn_model_type
        self.stream = stream
        self.config = config
        self.model = None
        self.bn_model = None
        self.preds = []
        self.risk_areas = None
        self.current_danger = 0
        self.current_pose = None

    #load yolo model for detection
    def load_model(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        xmsg('loading yolo model from .pt file.')
        self.model = YOLO(f'{xconst.DL_MODEL_ROOT}/{self.model_type.value}.pt')
        xmsg('yolo model loaded successfully.')
        return self.model
    
    #load binary classifier model
    def load_bn_model(self):
        xmsg('loading binary classifier model from .joblib file.')
        self.bn_model = joblib.load(f'{xconst.BN_MODEL_ROOT}/{self.bn_model_type.value}.joblib')
        xmsg('BN model loaded successfully.')
        return self.bn_model

    #add polygon to the prediction
    def set_risk_area(self, coords):
        self.risk_areas = coords
        self.load_bn_model()
        self.fallD = FallDetection(risk_areas=self.risk_areas, 
                                   bn_model=self.bn_model, 
                                   falling_criteria=FallConfig.FRAME_THRESHOLD)

    #predict the list of frames extracted from video file
    def detect(self, extract=True, save_file = False, filename=None):
        #load models if it is not loaded yet
        if not self.model: self.load_model()
        if not self.bn_model: self.load_bn_model()

        #extract the frames before prediction instread of realtime prediction
        if (self.stream.stream_type is StreamType.file) and extract:
            frames = self.stream.extract_frames()
            #iterate over all frames
            for idx, frame in enumerate(frames):
                yolo_result = self.model.track(frame, 
                                               verbose=False, 
                                               classes=xconst.DETECT_YOLO_CLASS, 
                                               conf=0.7)[0]
                #if result is not empty
                if len(yolo_result) > 0:
                    self.current_pose = yolo_result[0].keypoints.xy[0].cpu().numpy()
                    res = self.fallD.detect(result=XResult(yolo_result))
                    pred_plot = XPlot(result=res, config=xconst.plot_config).plot()

                    #add graph of features change
                    if xconst.ADD_GRAPH:
                        graph = self.fallD.bn_model.graph
                        pred_plot = cv2.hconcat([yolo_result[0].plot(), graph])

                    cv2.imshow('Prediction - Frame extracted', pred_plot)
                    self.preds.append(pred_plot)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        else:
            cap = self.stream.get_cap()
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    xerr('cannot read frame or video reach the end.')
                    break
                res = self.model(frame, verbose=False, classes=[0, 2])
                res = self.dangerD.detect(result=XResult(res))
                max_danger_level = max(res.danger_level)
                if max_danger_level != self.current_danger:
                    # xgpio.current_state = xgpio.gpio.LOW if max_danger_level > 0 else xgpio.gpio.HIGH
                    self.current_danger = max_danger_level
                pred = XPlot(result=res, config=xconst.plot_config).plot()
                pred = cv2.resize(pred, self.config.show_windows_size)
                polygon_color = (0, 0, 255) if max(res.danger_level) != 0 else (255, 0, 0)
                
                #update tower light image
                light_img_name = 'green_red_yellow_off'
                if max_danger_level > 0:
                    light_img_names = [xconst.pin_num_color[j] for j in [xconst.danger_pin_num[i] for i in xconst.danger_pin_num if i in res.danger_level]]
                    light_img_names.sort()
                    light_img_name = ['_'.join(light_img_names)][0]
                pred = cv2.hconcat([pred, self.tower_light_imgs[light_img_name]])
                self.preds.append(pred)
                cv2.imshow('Prediction - Realtime', pred)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cap.release()

        #clean the cv2 and gpio after prediction completed.
        self.model.cpu()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        cv2.destroyAllWindows()

        #save prediction into mp4 file
        if save_file:
            if not filename:
                unique_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{unique_id[:5]}_{timestamp}.mp4"
            self.save_prediction(filename)

    #save the prediction result into mp4 file
    def save_prediction(self, filename):
        #validate the stream type
        if self.stream.stream_type is not StreamType.file:
            xerr('only stream_type=file can be saved as a file.')
            return 0
        xmsg(f'wait - start saving prediction into file: "{filename}"')

        #prepare videowriter object
        _, height, width, _ = np.array(self.preds[:1]).shape
        codec_id = "mp4v" # ID for a video codec.
        fourcc = cv2.VideoWriter_fourcc(*codec_id)
        out = cv2.VideoWriter(os.path.join(xconst.PRED_SAVE_DIR, filename), fourcc=fourcc, fps=7, frameSize=(width, height))

        #write frames into file one by one
        xmsg('start writing frames into mp4 file.')
        start_time = time.time()
        for pred_frame in self.preds:
            out.write(pred_frame)
        out.release()

        #calculate saving time
        end_time = time.time()  # Record end time
        saving_time = end_time - start_time
        xmsg(f'file successfully saved as: "{filename}" | Saving time: {saving_time} seconds')