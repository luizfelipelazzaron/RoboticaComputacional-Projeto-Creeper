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

