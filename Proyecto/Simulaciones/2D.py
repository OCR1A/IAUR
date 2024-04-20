import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

#Parámetros del gripper
radio = 39.18               #Radio de la órbita en mm
offset_z = 340              #Profundidad de la cámara en mm
num_puntos = 100            
longitud_tangente = 11.5    #Distancia entre el centro del hardware y de la lente principal
xtool = 64
ytool = -269

#Parámetros del frame
ancho_rect = 565
alto_rect = 416

#Generar los puntos de la circunferencia
angulos = np.linspace(0, 2 * np.pi, num_puntos)
x = radio * np.cos(angulos) + xtool
y = radio * np.sin(angulos) + ytool
z = np.full_like(x, offset_z)

plt.figure()
plt.plot(x, y, label='Trayectoria')

#Magnitud del vector de posicion (xf, yf)
pos = np.sqrt(x**2 + y**2)

#Ángulo del vector de posición (xf, yf)
theta = np.arctan2(ytool, xtool)

#Coordenadas del punto de tangencia
wrist3 = 41.97
angulo_tangencia = theta + np.radians(45) - np.radians(wrist3) + np.radians(28)
x_tangencia = radio * np.cos(angulo_tangencia) + xtool
y_tangencia = radio * np.sin(angulo_tangencia) + ytool

print(f"Theta: {np.rad2deg(theta)}")

#Dirección de la tangente
angulo_tangente = angulo_tangencia - np.pi / 2  # Perpendicular al radio
x_final = x_tangencia + longitud_tangente * np.cos(angulo_tangente)
y_final = y_tangencia + longitud_tangente * np.sin(angulo_tangente)

#Componentes rectangulares del lente principal de la cámara
Ctcp = np.sqrt(radio**2 + longitud_tangente**2)
theta1 = np.arccos(radio / Ctcp)
theta2 = angulo_tangencia - theta1
cfx = Ctcp * np.cos(theta2) + xtool
cfy = Ctcp * np.sin(theta2) + ytool
rect = patches.Rectangle((cfx - ancho_rect/2, cfy - alto_rect/2), ancho_rect, alto_rect, linewidth=1, edgecolor='green', facecolor='none')

#Rotación del rectángulo
angulo_grados = np.degrees(angulo_tangente)
t = transforms.Affine2D().rotate_deg_around(cfx, cfy, angulo_grados) + plt.gca().transData
rect.set_transform(t)
plt.gca().add_patch(rect)

#Vector de posición del origen del frame
ofx = cfx - (ancho_rect/2) * np.cos(angulo_tangente) - (alto_rect/2) * np.sin(angulo_tangente)
ofy = cfy - (ancho_rect/2) * np.sin(angulo_tangente) + (alto_rect/2) * np.cos(angulo_tangente)

#Ejes unitarios
plt.arrow(xtool, ytool, 20*np.sin(theta), -20*np.cos(theta), head_width=5, head_length=5, fc='r', ec='r', label='Eje X unitario')
plt.arrow(xtool, ytool, 20*np.cos(theta), 20*np.sin(theta), head_width=5, head_length=5, fc='b', ec='b', label='Eje Y unitario')

#Posición de la herramienta
plt.plot([0, xtool], [0, ytool], label='Posición herramienta', color='orange')

#Recta tangente a la circunferencia
plt.plot([x_tangencia, x_final], [y_tangencia, y_final], label='Tangente', color='red')

#Vector de posición del origen del frame
plt.plot([0, ofx], [0, ofy], label='Origen frame', color='green')

#Marco de referencia de la cámara
plt.arrow(ofx, ofy, 20*np.cos(angulo_tangente), 20*np.sin(angulo_tangente), head_width=5, head_length=5, fc='r', ec='r', label='Eje X unitario')
plt.arrow(ofx, ofy, 20*np.sin(angulo_tangente), -20*np.cos(angulo_tangente), head_width=5, head_length=5, fc='b', ec='b', label='Eje Y unitario')

#Matrices de rotación del origen del frame
r = np.array([
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, 1]
])

R_z = np.array([
                [np.cos(angulo_tangencia), -np.sin(angulo_tangencia), 0],
                [np.sin(angulo_tangencia), np.cos(angulo_tangencia), 0],
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
    [200],
    [50],
    [0],
    [1]
])

coordTransf = np.dot(H0_f, punto)

print(coordTransf[0][0])
print(coordTransf[1][0])
print(ofx)
print(ofy)

plt.scatter(coordTransf[0][0], coordTransf[1][0], s=50, color='blue')

#Límites y etiquetas
plt.xlim([-100, 100])
plt.ylim([-100, 100])
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.axis('equal')
plt.grid('on')
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

#Mostrar el gráfico
plt.show()