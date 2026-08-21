#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
import cv2, pickle
from tqdm import tqdm
from modules import xconst
from .xenum import StreamType
from .xutils import xmsg, xerr
from modules.xcustom_class import Config

#class for managing stream
class Stream(object):
    def __init__(self, stream_type: StreamType, config: Config):
        self.stream_type = stream_type
        self.file_location = None
        self.cap = None
        self.config = config

    #set file for loading cap from (eg. video file)
    def set_file_location(self, loc):
        xmsg('target video file was set.')
        self.file_location = loc
    
    #get capture object
    def get_cap(self):
        if not self.cap: self.load_cap()
        return self.cap

    #load capture object using cv2
    def load_cap(self):
        xmsg('loading cv2 capture object.')
        #release cap if the app tries to reload the cap
        if self.cap: self.cap.release()
        if self.stream_type is StreamType.usb:
            self.cap = cv2.VideoCapture(0)
        elif self.stream_type is StreamType.file:
            if not self.file_location:
                xerr('please set file location before loading Cap.')
            else:
                self.cap = cv2.VideoCapture(self.file_location)
        xmsg('capture object loading completed.')
        return self.cap
    
    #check and validate cap
    def validate_cap(self):
        if not self.cap: self.load_cap()
        if not self.cap.isOpened():
            xerr('cap is not opened.')
            return False
        else:
            return True

    #display the captured frames
    def display(self, wh=(720, 480)):
        if not self.validate_cap(): return 0
        
        xmsg('start streaming video.')
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success:
                xerr('cannot read frame or video reach the end.')
                break
            
            frame = cv2.resize(frame, wh)
            cv2.imshow('Video Streaming', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    #extract frame from video file
    def extract_frames(self):
        if self.stream_type is not StreamType.file:
            xerr('only stream_type=file can extract frame from file.')
            return None
        else:
            if not xconst.DEVELOPMENT:
                if not self.validate_cap(): return 0
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                pbar = tqdm(total=total_frames, desc="Extracting frames", position=0, leave=True)
                extracted_frames = []
                cap = self.load_cap()
                xmsg(f'start extracting frames | total frames: {total_frames}')
                for frame_idx in range(total_frames):
                    success, frame = cap.read()
                    if not success: break
                    frame = cv2.resize(frame, self.config.cam_windows_size)
                    extracted_frames.append(frame)
                    pbar.update(1)

                # ==== start development
                # file_path = "./assets/videos/vid_001.pickle"
                # with open(file_path, 'wb') as f:
                #     pickle.dump(extracted_frames, f)
                # ==== end development
                return extracted_frames
            else:
                extracted_frames = []
                # ==== start development
                file_path = "./assets/videos/vid_001.pickle"
                # ==== end development
                with open(file_path, 'rb') as f:
                    extracted_frames = pickle.load(f)
                return extracted_frames