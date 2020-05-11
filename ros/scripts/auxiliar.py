# -*- coding:utf-8 -*-

import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

#'curl -F "file=@{}" https://api.anonfiles.com/upload'.format(str(nome_do_arquivo))


distancia = None

def to_1px(tpl):
    img = np.zeros((1,1,3), dtype=np.uint8)
    img[0,0,0] = tpl[0]
    img[0,0,1] = tpl[1]
    img[0,0,2] = tpl[2]
    return img

def to_hsv(value):
    # função que recebe value no formato [R,G,B]
    tpl = (value[0], value[1], value[2])
    hsv = cv2.cvtColor(to_1px( value ), cv2.COLOR_RGB2HSV)
    return hsv[0][0]

def ranges(value):
    # value = [R,G,B]
    hsv = to_hsv(value)
    hsv2 = np.copy(hsv)
    hsv[0] = max(0, hsv[0]-10)
    hsv2[0] = min(180, hsv[0]+ 10)
    hsv[1:] = 50
    hsv2[1:] = 255
    return hsv, hsv2 

def scaneou(dado):
    global distancia
    distancia = np.array(dado.ranges).round(decimals=2)[0]

# --------------------------------------------------------------------------------
#Funções do Projeto 2
# --------------------------------------------------------------------------------

colors = {
    "vermelho":(0,0,255),
    "azul":(255,0,0)
}

def auto_canny(image, sigma=0.33):
    """apply automatic Canny edge detection using the computed median"""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def coeficientes( array ):
    """Devolve uma tupla (m,n). m é o coeficiente angular e n é o coeficiente linear 
    da reta que representa o conjunto de retas do array"""
    if len(array) > 0:
        # array[:,t] -> coluna t da matriz chamada array
        x1, y1, x2, y2 = array[:,0], array[:,1], array[:,2], array[:,3]
        # m1 = (y2 - y1)/(x2-x1)
        m = (y2 - y1)/(x2 - x1)
        # n1 = m1*x1 + y1
        n = -m*x1 + y1
        
        m0 = round(np.median(m),2)
        n0 = round(np.median(n),2)
        return m0,n0
    
def tangente(linha):
    "Devolve a tangente do ângulo formado entre a linha e a horizontal (no referencial da imagem)"
    x1,y1,x2,y2 = linha[0]
    return float((y2-y1)/(x2-x1))

def draw_line(img, linha, color):
    color = str(color)
    color = colors[color]
    x1,y1,x2,y2 = linha[0]
    cv2.line(img, (x1, y1), (x2, y2), color, 2) 



