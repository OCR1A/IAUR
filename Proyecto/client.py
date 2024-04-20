#Communication between the robot and python
import socket

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

x = 0.2
y = -.7
z = 0.01

ur5_ip = "192.168.1.1" 
ur5_port = 30002 

ur5_move_command_1 = "movel(p[.064,-.269,.325,0,3.142,0], a = 1.2, v = 0.25, t = 0, r = 0)\n" #Origen 1 para pruebas
ur5_move_command_2 = "movel(p[.064,-.269,.325,2.471,1.940,0], a = 1.2, v = 0.25, t = 0, r = 0)\n" #Origen 2 para pruebas
ur5_move_command_3 = "movel(p[.064,-.269,.285,3.139,0.126,0], a = 1.2, v = 0.25, t = 0, r = 0)\n" #Origen 3 para pruebas
ur5_move_command_4 = "movel(p[-.060,-.270,.284,1.388,2.819,0], a = 1.2, v = 0.25, t = 0, r = 0)\n" #Origen 4 para pruebas
ur5_move_command_5 = "movel(p[-.174,-.286,.281,0.204,3.135,0], a = 1.2, v = 0.25, t = 0, r = 0)\n" #Origen 5 para pruebas
ur5_move_command_6 = "movel(p[.064,-.269,.400,0,3.142,0], a = 1.2, v = 0.25, t = 0, r = 0)\n" #Origen 6 para pruebas

send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_6)