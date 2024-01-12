import ultralytics
import torch
import torchvision
from ultralytics import YOLO

import os
import cv2
from ultralytics.utils.plotting import Annotator
import numpy as np
import time
from PIL import Image
import glob
import json


data_list = []
image_list = []
model_path = os.path.join('best.pt')
VIDEOS_DIR = os.path.join('.', 'videos')
IMG_DIR = os.path.join('.', 'img')
video_path = os.path.join('DJI_0300.MP4')


img_path = os.path.join('006783.jpg')
model = YOLO(model_path)
cap = cv2.VideoCapture(img_path)

# def write_json(new_data, filename):
#     with open(filename,'r+') as file:
#           # First we load existing data into a dict.
#         file_data = json.load(file)
#         # Join new_data with file_data inside emp_details
#         file_data["Defect"].append(new_data)
#         # Sets file's current position at offset.
#         file.seek(0)
#         # convert back to json.
#         json.dump(file_data, file, indent = 5)

start = False
while True:
    ret, img = cap.read()
    
    if not ret: 
        break
    
    # if mouse_clicked :
    #     print("click")
    #     while mouse_clicked :
    #         if cv2.waitKey(1) & 0xFF == ord(' '):
    #             break
    
    
    results = model.predict(img)
    
    

    for r in results:
        annotator = Annotator(img,font_size=0.1)
        
        boxes = r.boxes
        for box in boxes:
            print(f"boxSSSS: {box}")
            print(f"resultsType: {type(box)}")
            # box คือ แต่ละ กรอบ ที่มันจับได้ ใน boxes(ใหญ่)
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            Top = int(b[0])
            Left = int(b[1])
            Bottom = int(b[2])
            Right = int(b[3])
            h = Right - Left  # wide
            w = Bottom - Top  # high
            
            d = box.xywh[0]
            Toph = float(d[0])
            Lefth = float(d[1])
            Weighth = float(d[2])
            Heighth = float(d[3])

            f = box.xywhn[0]
            Xn = float(f[0])
            Yn = float(f[1])
            Weighthn = float(f[2])
            Heighthn = float(f[3])

            g = box.cls[0]
            clazz = int(g)

            c = box.cls
            #annotator.box_label(b)
            

            annotator.box_label(b, model.names[int(c)])
            
            
            print(f"{model.names[int(c)]},Top-Left: {(Top,Left)}, Bottom-Right: {(Bottom,Right)}")
            box_position_text1 = f"{Top,Left}" #tl
            box_position_text2 = f"{Bottom,Right}" #br
            box_position_text3 = f"{Top+w,Left}" #tr
            box_position_text4 = f"{Bottom-w,Right}" #bl
            print(f"{model.names[int(c)]},Xn-Yn: {(Xn,Yn)}, Weightn-Heightn: {(Weighthn,Heighthn)}")
            print(f"{model.names[int(c)]},Class: {(clazz)}")
            
            
            cv2.putText(img, box_position_text1, (Top-70, Left+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(img, box_position_text2, (Bottom+1, Right-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(img, box_position_text3, ((Top+w)+1, Left+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(img, box_position_text4, ((Bottom-w)-70, Right-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
             
            dictionary = {
                 "Class": clazz,
                 "X": Xn,
                 "Y": Yn,
                 "W": Weighthn,
                 "H": Heighthn
            }

            data_list.append(dictionary)

            # if type(out_file) is dict:
            #     data = [out_file]

            # out_file.append({
            #      "Class": clazz,
            #      "X": Xn,
            #      "Y": Yn,
            #      "W": Weighthn,
            #      "H": Heighthn
            #  })

            # out_file.close()

            

            # write_json(dictionary, file_name) 


    with open('number.json', 'w') as out_file:
        json.dump(data_list,out_file, indent=5)

    img = annotator.result()

    data_list.clear()

cap.release()

cv2.destroyAllWindows()


