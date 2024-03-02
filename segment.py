import cv2
import numpy as np

image = cv2.imread('instrumentos2.jpg')
image = cv2.resize(image, (1020, 720))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Coordenadas de cada vertice, pendientes e intersección con y
vertices = []
m = []
b = []
xi = [] #Guarda la iésima x
yi = [] #Guarda la iésima y

# Ajustar el contraste y el brillo
alpha = 1  # Contraste
beta = -250  # Brillo
adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
adjusted_image = np.where(adjusted_image <= 110, 0, adjusted_image)
adjusted_image = np.where(adjusted_image > 110, 255, adjusted_image)

#Pasa a escala de grises
gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2GRAY)

#Encuentra los contornos
contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Encuentra el contorno más grande (que debería ser el contorno del área blanca)
largest_contour = max(contours, key=cv2.contourArea)

#Aproximar el contorno para hacer que los bordes sean más angulares
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
#Ordenar los puntos de approx de izquierda a derecha
#approxsorted = sorted(approx, key=lambda x: x[0][0])

#Obtenemos las coordenadas de cada intersección
for point in approx:
    x, y = point[0]
    vertices.append((x, y))

#Almacenamos xi, yi en arreglos individuales
for i in range(0, len(vertices)):
    xi.append(vertices[i][0])
for vertice in vertices:
    yi.append(vertice[1])

for i in range(0, 7, 2):
    m_line = (yi[i+1] - yi[i]) / (xi[i+1] - xi[i])
    m.append(m_line)
    b_line = yi[i] - m_line * xi[i]
    b.append(b_line)

#Elimina los elementos impares de la lista
for i in range(len(xi)-1, -1, -2):
    xi.pop(i)
    yi.pop(i)

for i in range(len(m)-1):
    #Puntos extremos originales de la línea
    x1 = xi[i]
    y1 = yi[i]
    x2 = xi[i+1]
    y2 = yi[i+1]
    
    #Calcular puntos adicionales para extender la línea
    extended_length = 1000  # Longitud de la extensión, puedes ajustar esto según sea necesario
    x1_extended = int(x1 - extended_length)
    y1_extended = int(m[i] * x1_extended + b[i])
    x2_extended = int(x2 + extended_length)
    y2_extended = int(m[i] * x2_extended + b[i])
    
    #Dibujar la línea extendida
    cv2.line(image, (x1_extended, y1_extended), (x2_extended, y2_extended), (0, 0, 0), 5)
    
#Línea con coordenadas, ni[3], ni[0]
x1 = xi[0] 
x2 = xi[3]
y1 = yi[0]
y2 = yi[3]
extended_length = 1000
x1_extended = int(x1 - extended_length)
y1_extended = int(m[3] * x1_extended + b[3])
x2_extended = int(x2 + extended_length)
y2_extended = int(m[3] * x2_extended + b[3])

#Dibujar la línea especial extendida
cv2.line(image, (x1_extended, y1_extended), (x2_extended, y2_extended), (0, 0, 0), 5)

cv2.imshow('Resultado', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
