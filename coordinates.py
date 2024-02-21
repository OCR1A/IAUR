import cv2 
from ultralytics import YOLO
import numpy as np
import os
def arduino_map(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

cam_num = 1
pre_model = r'Med_IA.pt'
flag = False
activate = False
cam = cv2.VideoCapture(cam_num)
model = YOLO(pre_model)

def calculateCenter(image, center_x, center_y, red, green, blue):
    cv2.circle(image, (center_x, center_y), 5, (blue, green, red), -1)  # Draw a filled circle

def foto():
        
    #Configuración inicial de cámara
    cam = cv2.VideoCapture(cam_num)
    success, frame = cam.read()

    if success:
        #Convierte el formato de color de BGR a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Redimensiona la imagen según sea necesario
        resized_frame = cv2.resize(frame, (1020, 720))

        #Aplica el modelo a la imagen con una confianza del 0.7 y clases específicas
        results = model(resized_frame, conf=0.7, classes=[0, 2, 3, 4])

        #Obtiene la imagen con las anotaciones del modelo
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        for box in boxes:
            if 0 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Bisturí x pixels: {center_x}")
                print(f"Bisturí y pixels: {center_y}")
                print(f"Bisturí x cm: {arduino_map(center_x, 0, 1020, 0, 38.4)}")
                print(f"Bisturí y cm: {arduino_map(center_y, 0, 720, 0, 28)}")
                calculateCenter(annotated_frame,center_x, center_y, 31, 112 ,255)
            elif 2 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Pinzas x: {center_x}")
                print(f"Pinzas y: {center_y}")
                print(f"Pinzas x cm: {arduino_map(center_x, 0, 1020, 0, 38.4)}")
                print(f"Pinzas y cm: {arduino_map(center_y, 0, 720, 0, 28)}")
                calculateCenter(annotated_frame,center_x, center_y, 31, 112 ,255)
            elif 3 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Tijeras curvas x: {center_x}")
                print(f"Tijeras curvas y: {center_y}")
                print(f"Tijeras curvas x cm: {arduino_map(center_x, 0, 1020, 0, 38.4)}")
                print(f"Tijeras curvas y cm: {arduino_map(center_y, 0, 720, 0, 28)}")
                calculateCenter(annotated_frame,center_x, center_y, 31, 112 ,255)
            elif 4 in box.cls:
                coordinates = box.xyxy.squeeze().tolist()
                center_x = int((coordinates[0] + coordinates[2]) / 2)
                center_y = int((coordinates[1] + coordinates[3]) / 2)
                print(f"Tijeras rectas x: {center_x}")
                print(f"Tijeras rectas y: {center_y}")
                print(f"Tijeras rectas x cm: {arduino_map(center_x, 0, 1020, 0, 38.4)}")
                print(f"Tijeras rectas y cm: {arduino_map(center_y, 0, 720, 0, 28)}")
                calculateCenter(annotated_frame,center_x, center_y, 31, 112 ,255)

        #UR debug
        user_dir = os.path.expanduser(r'C:\Users\Salvador\Desktop')
        photo_filename = "anotacion_modelo.jpg"
        photo_path = os.path.join(user_dir, photo_filename)
        cv2.imwrite(photo_path, annotated_frame)
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        print("La imagen se ha guardado en:", photo_path)
    
    cam.release()

foto()