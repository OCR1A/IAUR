import socket
import struct
import pyrealsense2 as rs
import cv2
import numpy as np

def mapValue(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        #Guarda las coordenadas del clic en la lista de coordenadas
        params['coords'].append((x, y))

def getPose():
    #Conexión con el UR5
    HOST = '192.168.1.1'
    PORT = 30002

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect((HOST, PORT))

    cdr = 0
    while cdr < 3:
        #Recibir 4096 bytes de datos (suficiente para almacenar cualquier paquete)
        data = s.recv(4096)
        #Inicializar i para trackear la posición en el paquete
        i = 0
        if data:
            #Información del paquete recibido
            packlen = (struct.unpack('!i', data[0:4]))[0]
            timestamp = (struct.unpack('!Q', data[10:18]))[0]
            packtype = (struct.unpack('!b', data[4:5]))[0]

            #Si el tipo de paquete es el estado del robot, loopear hasta alcanzar el final del paquete
            if packtype == 16:
                while i + 5 < packlen:
                    #Tamaño y tipo de mensaje
                    msglen = (struct.unpack('!i', data[5+i:9+i]))[0]
                    msgtype = (struct.unpack('!b', data[9+i:10+i]))[0]

                    if msgtype == 1:
                        #Si el mensaje es coordenada articular, crear una lista para guardar los ángulos
                        angle = [0] * 6
                        j = 0
                        while j < 6:
                            #Bytes del 10 al 18 tienen ángulo j0, cada coordenada articular es de 41 bytes (saltamos j*41 cada vez)
                            angle[j] = (struct.unpack('!d', data[10+i+(j*41):18+i+(j*41)]))[0]
                            j += 1

                    #Si el tipo de mensaje son coordenadas cartesianas, almacenamos las coordenadas actuales del TCP
                    elif msgtype == 4:
                        x = (struct.unpack('!d', data[10+i:18+i]))[0]
                        y = (struct.unpack('!d', data[18+i:26+i]))[0]
                        z = (struct.unpack('!d', data[26+i:34+i]))[0]
                        rx = (struct.unpack('!d', data[34+i:42+i]))[0]
                        ry = (struct.unpack('!d', data[42+i:50+i]))[0]
                        rz = (struct.unpack('!d', data[50+i:58+i]))[0]

                    #Incrementamos i por el tamaño del mensaje para movernos al siguiente mensaje en el paquete
                    i += msglen
        cdr = cdr + 1

    return x, y, z, rx, ry, rz, angle[5]

def send_instruction_to_ur5(ip, port, instruction):
    try:
        #Socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        #Connect to robot
        sock.connect((ip, port))
        
        #Send the instruction to the robot
        sock.sendall(instruction.encode())
        
        print("Instruction sent to UR5 robot:", instruction)
        
        #Close the socket
        sock.close()
        
    except ConnectionRefusedError:
        print("Connection refused")
    
    except TimeoutError:
        print("Timeout, can't connect")

def iniciar_camara():
    #Inicializar la cámara RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            #Esperar a que haya datos disponibles
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            #Convertir los frames a matrices numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            #Dibujar un punto en el centro del frame de color
            height, width, _ = color_image.shape
            center_x, center_y = int(width / 2), int(height / 2)
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

            #Obtener la distancia al punto central
            distance = depth_frame.get_distance(center_x, center_y)

            #Calcular distancias horizontales y verticales basadas en la distancia
            H = distance*np.tan(23) #23
            V = distance*np.tan(83.96) 
            V = np.abs(V)

            #Mostrar la imagen con el punto dibujado y la distancia
            cv2.putText(color_image, f"Distance: {distance:.2f} m, H: {H:.2f} m, V: {V:.2f} m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('RealSense Camera', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def capturar_y_mostrar_foto():
    #Configurar el pipeline y el stream de la cámara RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    clic_coords = {'coords': []}

    try:
        #Esperar a que haya un frame disponible
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("No se pudo obtener el frame de color.")
            return

        #Convertir el frame de color a un arreglo de NumPy
        color_image = np.asanyarray(color_frame.get_data())

        #Calcular el centro de la imagen y dibujar un círculo rojo
        height, width, _ = color_image.shape
        center_x, center_y = int(width / 2), int(height / 2)
        cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

        #Obtener la distancia en el centro y calcular dimensiones H y V
        distance = depth_frame.get_distance(center_x, center_y)
        H = distance*np.tan(23) #23
        V = distance*np.tan(83.96) 
        V = np.abs(V)

        #Conversión a mm
        V = V * 1000
        H = H * 1000

        #Mostrar la imagen con el punto dibujado y la distancia
        cv2.putText(color_image, f"Distance: {distance:.2f} m, H: {H:.2f} m, V: {V:.2f} m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        #Mostrar la imagen anotada
        cv2.imshow('RealSense Camera Photo', color_image)
        cv2.setMouseCallback('RealSense Camera Photo', mouse_callback, clic_coords)

        #Esperar a que el usuario presione una tecla antes de cerrar la ventana
        cv2.waitKey(0)
    finally:
        #Detener la captura y cerrar todas las ventanas
        pipeline.stop()
        cv2.destroyAllWindows()

        return H, V, clic_coords['coords']

def frameOriginCoordinates(xtool, ytool, H, V, wrist3):
    radio = 39.18
    longitud_tangente = 11.5

    #Ángulo del vector de posición (xf, yf)
    theta = np.arctan2(ytool, xtool)

    #Coordenadas del punto de tangencia
    angulo_tangencia = theta + np.radians(45) - np.radians(wrist3) + np.radians(28)

    #Dirección de la tangente
    angulo_tangente = angulo_tangencia - np.pi / 2

    #Componentes rectangulares del lente principal de la cámara
    Ctcp = np.sqrt(radio**2 + longitud_tangente**2)
    theta1 = np.arccos(radio / Ctcp)
    theta2 = angulo_tangencia - theta1
    cfx = Ctcp * np.cos(theta2) + xtool
    cfy = Ctcp * np.sin(theta2) + ytool

    #Vector de posición del origen del frame
    ofx = cfx - (H/2) * np.cos(angulo_tangente) - (V/2) * np.sin(angulo_tangente)
    ofy = cfy - (H/2) * np.sin(angulo_tangente) + (V/2) * np.cos(angulo_tangente)

    return ofx, ofy, angulo_tangencia

def transformCoordinates(x1, y1, ofx, ofy, theta):
    r = np.array([
                    [0, -1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]
    ])

    R_z = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
    ])

    R0_f = np.dot(R_z, r)

    P0_f = np.array([
                        [ofx],
                        [ofy],
                        [0]
    ])

    #Poderosas matrices de transformación homogenea
    H0_f = np.concatenate((R0_f, P0_f), 1)
    H0_f = np.concatenate((H0_f, [[0,0,0,1]]), 0)

    punto = np.array([
        [x1],
        [y1],
        [0],
        [1]
    ])

    coordTransf = np.dot(H0_f, punto)
    
    return coordTransf[0][0], coordTransf[1][0]

#Configuración del socket
ur5_ip = "192.168.1.1" 
ur5_port = 30002 

#Parámetros para obtener las coordenadas cartesianas del origen del frame
xr, yr, zr, rxr, ryr, rzr, wrist3r = getPose()
xr = xr * 1000
yr = yr * 1000
Hr, Vr, objCoor = capturar_y_mostrar_foto()
angle = np.rad2deg(wrist3r)
ofxr, ofyr, thetar = frameOriginCoordinates(xr, yr, Hr, Vr, angle)
xfinal = mapValue(objCoor[0][0], 0, 640, 0, Hr)
yfinal = mapValue(objCoor[0][1], 0, 480, 0, Vr)
xtransf, ytransf = transformCoordinates(xfinal, yfinal, ofxr, ofyr, thetar)

#Conversión de unidades
ofxr = ofxr / 1000
ofyr = ofyr / 1000
xtransf = xtransf / 1000
ytransf = ytransf / 1000
ofzr = 0.05
print(f"ofxr: {ofxr}")
print(f"ofyr: {ofyr}")
print(f"ofzr: {ofzr}")
print(f"rx: {rxr}")
print(f"ry: {ryr}")
print(f"rz: {rzr}")
print(objCoor)
print(f"Objeto en x: {objCoor[0][0]}")
print(f"Objeto en y: {objCoor[0][1]}")
print(f"xtransf: {xtransf}")
print(f"ytransf: {ytransf}")

ur5_move_command = f"movel(p[{xtransf},{ytransf},{ofzr},{rxr},{ryr},{rzr}], a = 1.2, v = 0.25, t = 0, r = 0)\n"
send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command)

#iniciar_camara()