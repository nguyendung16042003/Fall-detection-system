#==============================#
# Author: KONG Vungsovanreach  #
# Date: 2024-04-25             #
#==============================#

#import all required modules
import cv2
from .xenum import BBType
from .xutils import xmsg, xerr, isnn
from shapely.geometry import Polygon, Point

#base class for plotting result
class XPlot(object):
    def __init__(self, result, config, risk_areas):
        self.result = result
        self.config = config
        self.risk_areas = Polygon(risk_areas[0])

    @property
    def boxes(self): #return all predicted boxes
        return self.result.boxes

    @property
    def xboxes(self):
        return self.result.xboxes

    @property
    def img(self): #return original input image
        return self.result.img
    
    #plot bounding box around each object
    def add_bb(self, p1, p2, bc, bb_type=BBType.CORNER, line_length=30):
        (x1, y1), (x2, y2), (b_xc, b_yc) = p1, p2, bc
        if bb_type is BBType.CORNER:
            #top left
            cv2.line(self.img, (x1, y1), (x1, y1+line_length), self.config.color.bb, thickness=self.config.thickness.bb)
            cv2.line(self.img, (x1, y1), (x1+line_length, y1), self.config.color.bb, thickness=self.config.thickness.bb)
            #top right    
            cv2.line(self.img, (x2, y1), (x2, y1+line_length), self.config.color.bb, thickness=self.config.thickness.bb)
            cv2.line(self.img, (x2, y1), (x2-line_length, y1), self.config.color.bb, thickness=self.config.thickness.bb)
            #bottom right
            cv2.line(self.img, (x2, y2), (x2, y2-line_length), self.config.color.bb, thickness=self.config.thickness.bb)
            cv2.line(self.img, (x2, y2), (x2-line_length, y2), self.config.color.bb, thickness=self.config.thickness.bb)
            #bottom left
            cv2.line(self.img, (x1, y2), (x1, y2-line_length), self.config.color.bb, thickness=self.config.thickness.bb)
            cv2.line(self.img, (x1, y2), (x1+line_length, y2), self.config.color.bb, thickness=self.config.thickness.bb)

        elif bb_type is BBType.ELLIPSE:
            cv2.ellipse(self.img, (b_xc, b_yc), (15, 15), 0, 20, 160, self.config.color.bb, self.config.thickness.bb)
            cv2.ellipse(self.img, (b_xc, b_yc), (10, 10), 0, 20, 160, self.config.color.bb, self.config.thickness.bb)

        elif bb_type is BBType.NORMAL:
            cv2.rectangle(self.img, (x1, y1), (x2, y2), self.config.color.bb, thickness=self.config.thickness.bb, lineType=cv2.LINE_AA)
    
    #plot class name of each object on frame
    def add_cls_name(self, cls_name, p1):
        cv2.putText(self.img, cls_name, (p1[0]-2, p1[1]-10), self.config.font.all, self.config.font_scale.cls_name, self.config.color.cls_name, self.config.thickness.cls_name)

    #plot lying percentage from history
    def add_lying_percent(self, lying_percent, pc):
        cv2.putText(self.img, f'{lying_percent}%', (pc[0], pc[1]), self.config.font.all, self.config.font_scale.lying_percent, self.config.color.cls_name, self.config.thickness.lying_percent)

    def plot(self):
        #iterate over all boxes and plot detail
        for idx, xbox in enumerate(self.xboxes):
            track_id, lying_percent = None, 0
            try:
                x1, y1, x2, y2 = self.result.xyxy[idx]
                x, y, w, h = self.result.xyxy[idx]
                xc, yc = self.result.xcyc[idx]
                b_xc, b_yc = self.result.bottom_xcyc[idx]
                class_id = self.result.cls[idx]
                class_name = self.result.names[class_id]

                if isnn(self.result.ids): 
                    track_id = self.result.ids[idx]
                    lying_percent = self.result.data.history.data[track_id].lying_percent

                #plot things
                if self.config.bb: #plot bounding box
                    if self.config.only_in_risk_area: #only plot falls in polygon
                        if Point((xc, yc)).within(self.risk_areas): #if falls occures in the polygon
                            self.add_bb((x1, y1), (x2, y2), (b_xc, b_yc), bb_type=self.config.bb_type, line_length=15) #plot bounding box
                            cv2.circle(self.img, (xc, yc), radius=2, color=(0, 255, 0), thickness=-1)
                    else:
                        self.add_bb((x1, y1), (x2, y2), (b_xc, b_yc), bb_type=self.config.bb_type, line_length=15) #plot bounding box

                if self.config.cls_name: self.add_cls_name(class_name, (x1, y1))
                if self.config.lying_percent: self.add_lying_percent(lying_percent, (xc, yc))
            except Exception as e:
                pass
                # xerr(e)
                # xerr('failed to extract object information such as ID,...')
        if sum(self.result.data.fall) > 0:
            if self.config.fall_alert:
                cv2.rectangle(self.img, (0, 0), (250, 35), self.config.color.fall_alert_bg, -1)
                cv2.putText(self.img, f'Fall is detected.', (10, 25), self.config.font.all, self.config.font_scale.fall_alert_text, self.config.color.fall_alert_text, self.config.thickness.fall_alert_text)
        return self.img