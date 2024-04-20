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

send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_1)

# Application examples:
# "movel(p[.15,-.6,-.392,3.053,.747,-.], a = 1.2, v = 0.25, t = 0, r = 0)\n"
# send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_1)
# time.sleep(5)
# send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_3)
# time.sleep(5)
# send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_4)
# time.sleep(3)
# send_instruction_to_ur5(ur5_ip, ur5_port, ur5_move_command_5)
# time.sleep(5)
# movej command ([]):
# movej([theta1, theta2, theta3, theta4, theta5, theta6, a, v, t, r]) 
# theta1...theta6 are the angles in radians for each joint
# a: System acceleration
# v: System velocity
# t: Travel time between coordinates, if greater than 0 it overrides a and v
# r: Blend/smoothing of the movement as it approaches the target x,y,z
# Example: "movej([0, -1.5708, -1.5708, -1.5708, 1.5708, 3.1416], a=1, v=1.05)\n"

# movel command ([]):
# movel(p[X,Y,Z,RX,RY,RZ], a, v, t, r)
# X,Y,Z are the desired coordinates of the end effector in cartesian space given in meters
# RX,RY,RZ are the rotations of the end effector's reference frame with respect to the fixed (base/world) frame given in radians
# a: System acceleration
# v: System velocity
# t: Travel time between coordinates, if greater than 0 it overrides a and v
# r: Blend/smoothing of the movement as it approaches the target x,y,z
# Example: "movel(p[.34,-.6,.7,1.382,.633,.653], a = 1.2, v = 0.25, t = 0, r = 0)\n"
# NOTE: It's very important to use the format shown in the example.

# actual_tcp_pose = get_actual_tcp_pose()
# rotation_matrix = get_rotation(actual_tcp_pose)
