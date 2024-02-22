import socket
import time

def send_instruction_to_ur5(ip, port, instruction):
    try:
        #Objeto socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        #Conectar al robot
        sock.connect((ip, port))
        
        #Manda la instrucción al robot
        sock.sendall(instruction.encode())
        
        print("Instruction sent to UR5 robot:", instruction)
        
        #Cierra el socket
        sock.close()
        
    except ConnectionRefusedError:
        print("Conexión rechazada, no se pudo conectar")
    
    except TimeoutError:
        print("Tiempo de espera agotado, no se pudo conectar")

x = 0.2
y = -.7
z = 0.01

ur5_ip = "192.168.1.1" 
ur5_port = 30002 

ur5_move_command_1 = "movel(p[.34,-.6,0.7,1.382,.633,.653], a = 1.2, v = 0.25, t = 0, r = 0)\n"
ur5_move_command_2 = "movej([0, -1.5708, -1.5708, -1.5708, 1.5708, 3.1416], a=1, v=1.05)\n"
ur5_move_command_3 = "movel(p[.34,-.6,0.01,3.1415,0,0], a = 1.2, v = 0.25, t = 0, r = 0)\n"
ur5_move_command_4 = f"movel(p[{x},{y},{z},3.1415,0,0], a = 1.2, v = 0.25, t = 0, r = 0)\n"
ur5_move_command_5 = "movel(p[-.14,-.6,0.01,3.1415,0,0], a = 1.2, v = 0.25, t = 0, r = 0)\n"

send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_2)
#"movel(p[.15,-.6,-.392,3.053,.747,-.], a = 1.2, v = 0.25, t = 0, r = 0)\n"
#send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_1)
#time.sleep(5)
#send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_3)
#time.sleep(5)
#send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_4)
#time.sleep(3)
#send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_5)
#time.sleep(5)
#Comando movej([]):
#movej([theta1, theta2, theta3, theta4, theta5, theta6, a, v, t, r]) 
#theta1...theta6 son los ángulos en radianes de cada joint
#a: Aceleración del sistema
#v: Aceleración del sistmea
#t: Tiempo de viaje entre coordenadas, si es mayor a 0 anula a y v
#r: Blend/suavizado del movimiento al acercarse al x,y,z de destino
#Ejemplo: "movej([0, -1.5708, -1.5708, -1.5708, 1.5708, 3.1416], a=1, v=1.05)\n"

#Comando movel([])
#movel(p[X,Y,Z,RX,RY,RZ], a, v, t, r)
#X,Y,Z Son las coordenadas deseadas del efector final en el espacio cartesiano dadas en metros
#RX,RY,RZ Son las rotaciones del marco de referencia del efector final respecto al marco fijo (base/mundo) dadas en radianes
#a: Aceleración del sistema
#v: Aceleración del sistmea
#t: Tiempo de viaje entre coordenadas, si es mayor a 0 anula a y v
#r: Blend/suavizado del movimiento al acercarse al x,y,z de destino
#Ejemplo: "movel(p[.34,-.6,.7,1.382,.633,.653], a = 1.2, v = 0.25, t = 0, r = 0)\n"
#NOTA: Es muy importante utilizar el formato mostrado en el ejemplo.

#actual_tcp_pose = get_actual_tcp_pose()
#rotation_matrix = get_rotation(actual_tcp_pose)
