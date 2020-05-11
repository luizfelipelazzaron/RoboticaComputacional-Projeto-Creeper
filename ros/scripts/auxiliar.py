# -*- coding:utf-8 -*-

from __future__ import print_function, division
import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

#'curl -F "file=@{}" https://api.anonfiles.com/upload'.format(str(nome_do_arquivo))


distancia = None


def to_1px(tpl):
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0, 0] = tpl[0]
    img[0, 0, 1] = tpl[1]
    img[0, 0, 2] = tpl[2]
    return img


def to_hsv(value):
    # função que recebe value no formato [R,G,B]
    tpl = (value[0], value[1], value[2])
    hsv = cv2.cvtColor(to_1px(value), cv2.COLOR_RGB2HSV)
    return hsv[0][0]


def ranges(value):
    # value = [R,G,B]
    hsv = to_hsv(value)
    hsv2 = np.copy(hsv)
    hsv[0] = max(0, hsv[0]-10)
    hsv2[0] = min(180, hsv[0] + 10)
    hsv[1:] = 50
    hsv2[1:] = 255
    return hsv, hsv2


def scaneou(dado):
    global distancia
    distancia = np.array(dado.ranges).round(decimals=2)[0]


def draw_line(img, linha, color):
    """Dada uma linha do tipo `[[x0, y0, x1, y1]]`, retorna desenha elas na imagem `img` e na cor `color`;"""
    # color = colors[color]
    x1, y1, x2, y2 = linha[0]
    cv2.line(img, (x1, y1), (x2, y2), color, 2)


def tangente(linha):
    "Devolve a tangente do ângulo formado entre a linha e a horizontal (no referencial da imagem)"
    # print("linha.shape:",linha.shape)
    # print("linha:",linha)
    x1, y1, x2, y2 = linha[0]
    tangente = (y2-y1)/(x2-x1)
    # print("tangente:",tangente) # print para ajudar a debugar
    return tangente


def coefficients(array):
    """Devolve uma tupla (m,n). m é o coeficiente angular e n é o coeficiente linear 
    da reta que representa o conjunto de retas do array"""
    if len(array) > 0:
        # array[:,t] -> coluna t da matriz chamada array
        x1, y1, x2, y2 = array[:, :, 0], array[:,
                                               :, 1], array[:, :, 2], array[:, :, 3]
        # m1 = (y2 - y1)/(x2-x1)
        m = (y2 - y1)/(x2 - x1)
        # n1 = m1*x1 + y1
        n = -m*x1 + y1

        m = round(np.median(m), 2)
        n = round(np.median(n), 2)
        # print("(m,n)") # print pra ajudar a debugar
        return m, n


def cross(img, x, y):
    # x = int(x)
    # y = int(y)
    cv2.line(img, (x-10, y), (x+10, y), (255,32,255), 2)
    cv2.line(img, (x, y-10), (x, y+10), (255, 32, 255), 2)

def line(img, x0, y0, x1, y1):
    cv2.line(img, (x0,y0), (x1,y1), (255,32,255), 2)

def drawHUD(img, x, y, tolerance):
    x = int(x)
    y = int(y)
    cross(img,x,y)
    line(img, x - tolerance, y - 20, x - tolerance, y+20)
    line(img, x + tolerance, y - 20, x + tolerance, y+20)
