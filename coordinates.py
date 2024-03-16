#Main code
import cv2 
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
import os
def mapValue(value, from_low, from_high, to_low, to_high):
    #Rescale one range of numbers to another
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def foto(cam_num):
        
    #Normal camera init (Not RealSense camera)
    cam = cv2.VideoCapture(cam_num)
    success, frame = cam.read()

    if success:
        #From BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Image resize
        resized_frame = cv2.resize(frame, (1020, 720))

        #Model application with confidence rate of 0.7 applied to all classes excepting class 1
        results = model(resized_frame, conf=0.7, classes=[0, 2, 3, 4])

        #Stores the image with the model annotations
        annotated_frame = results[0].plot()

        #Object's (x, y) coordinates
        boxes = results[0].boxes
        for box in boxes:
            if 0 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Scalpel 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}m")
                print(f"Scalpel 'y': {mapValue(center_y, 0, 720, 0, 28)}m")
                cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
            elif 2 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Pliers 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}")
                print(f"Pliers 'y': {mapValue(center_y, 0, 720, 0, 28)}")
                cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
            elif 3 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Curved scissors 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}m")
                print(f"Curved scissors 'y': {mapValue(center_y, 0, 720, 0, 28)}m")
                cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
            elif 4 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Straigh scissors 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}m")
                print(f"Straigh scissors 'y': {mapValue(center_y, 0, 720, 0, 28)}m")
                cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)

        #Saves the image locally
        user_dir = os.path.expanduser(r'C:\Users\Salvador\Desktop')
        photo_filename = "model_annotation.jpg"
        photo_path = os.path.join(user_dir, photo_filename)
        cv2.imwrite(photo_path, annotated_frame)
        print("The image was stored:", photo_path)
    
    cam.release()

def showCamera(model, cam_num):
    #Variable to store the camera data
    cam = cv2.VideoCapture(cam_num) 

    while True:

        #Capture frame by frame
        success, frame = cam.read()

        if success:
            resized_frame = cv2.resize(frame, (1020, 720))
            results = model(resized_frame, conf=0.7, classes=[0, 2, 3, 4])
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            for box in boxes:
                if 0 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Scalpel 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}m")
                    print(f"Scalpel 'y': {mapValue(center_y, 0, 720, 0, 28)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
                elif 2 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Pliers 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}m")
                    print(f"Pliers 'x': {mapValue(center_y, 0, 720, 0, 28)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
                elif 3 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Curved scisors 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}m")
                    print(f"Curved scisors 'y': {mapValue(center_y, 0, 720, 0, 28)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
                elif 4 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Right scisors 'x': {mapValue(center_x, 0, 1020, 0, 38.4)}m")
                    print(f"Right scisors 'y': {mapValue(center_y, 0, 720, 0, 28)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)

            cv2.imshow('Result', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Unaible to capture the camera's frame")
            break

    cam.release()
    cv2.destroyAllWindows()

def showCameraRealSense(model):
    #RealSense d455 camera pipeline and settings
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            height, width, _ = color_image.shape
            center_x, center_y = int(width / 2), int((height / 2))
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

            distance = depth_frame.get_distance(center_x, center_y)

            H = distance*np.tan(23) 
            V = np.abs(distance*np.tan(83.96)) 

            #Shows the image with the desired measurings
            cv2.putText(color_image, f"Distance: {distance:.2f} m, H: {H:.2f} m, V: {V:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('RealSense Camera', color_image)

            resized_frame = cv2.resize(color_image, (1020, 720))
            
            #Applies the detection model
            results = model(resized_frame, conf=0.7, classes=[0, 2, 3, 4])
            annotated_frame = results[0].plot()

            #Processing the model results
            boxes = results[0].boxes
            for box in boxes:
                if 0 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Scalpel in 'x': {mapValue(center_x, 0, 1020, 0, H)}m")
                    print(f"Scalpel in 'y': {mapValue(center_y, 0, 720, 0, V)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
                elif 2 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Pliers in 'x': {mapValue(center_x, 0, 1020, 0, H)}m")
                    print(f"Pliers in 'y': {mapValue(center_y, 0, 720, 0, V)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
                elif 3 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Curved scisors in 'x': {mapValue(center_x, 0, 1020, 0, H)}m")
                    print(f"Curved scisors in 'y': {mapValue(center_y, 0, 720, 0, V)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)
                elif 4 in box.cls:
                    coordinates = box.xyxy.squeeze().tolist()
                    center_x = int((coordinates[0] + coordinates[2]) / 2)
                    center_y = int((coordinates[1] + coordinates[3]) / 2)
                    print(f"Right scisors in 'x': {mapValue(center_x, 0, 1020, 0, H)}m")
                    print(f"Right scisors in 'y': {mapValue(center_y, 0, 720, 0, V)}m")
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (31, 112, 255), -1)

            #Showing the annotated image
            cv2.imshow('Resultado', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        #Stop the capture and close all windows
        pipeline.stop()
        cv2.destroyAllWindows()

#Settings and RealSense D455 init.
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
 
model = YOLO("Med_IA.pt")
showCameraRealSense(model)
