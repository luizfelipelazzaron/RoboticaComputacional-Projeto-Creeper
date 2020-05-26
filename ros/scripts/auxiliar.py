# -*- coding:utf-8 -*-

from __future__ import print_function, division
import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

#'curl -F "file=@{}" https://api.anonfiles.com/upload'.format(str(nome_do_arquivo))


distancia = None
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
HUD_COLOR = (32,255,0)
# HUD_COLOR = (200,32,100)

def to_1px(tpl):
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0, 0] = tpl[0]
    img[0, 0, 1] = tpl[1]
    img[0, 0, 2] = tpl[2]
    return img


def to_hsv(value):
    """função que recebe value no formato [R,G,B]"""
    tpl = (value[0], value[1], value[2])
    hsv = cv2.cvtColor(to_1px(value), cv2.COLOR_RGB2HSV)
    return hsv[0][0]


def ranges(value):
    """value = [R,G,B]"""
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
    """cross(img, x, y) recebe uma imagem (array do OpenCv),um x (float) e y(float)"""
    x = int(x)
    y = int(y)
    line(img, x-10, y, x+10, y)
    line(img, x, y-10, x, y+10)
    return img

def line(img, x0, y0, x1, y1):
    cv2.line(img, (x0,y0), (x1,y1), (0,0,0), 2)
    cv2.line(img, (x0,y0), (x1,y1), HUD_COLOR, 1)

def text(img, text, x0, y0):
    cv2.putText(img=img,text=text, org=(x0,y0), fontFace=1, fontScale=1, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=img,text=text, org=(x0,y0), fontFace=1, fontScale=1, color=HUD_COLOR, thickness=1, lineType=cv2.LINE_AA)
    return img

def textLines(img, array, tolerancia, position, altura=None, largura=None):
    if position == "topLeft":
        for index, row in enumerate(array):
            key, value = row
            img = text(img, "{0}: {1}".format(key,value), x0=tolerancia + 10, y0=tolerancia + 20 + index*20)

    elif position == "bottomLeft":
        for index, row in enumerate(array):
            key, value = row
            img = text(img, "{0}: {1}".format(key,value), x0=tolerancia + 10, y0=altura - tolerancia - 20*(len(array)-index))

    return img

def drawHUD(img, instance):
    """
    Desenha marcações importantes na tela;
    """
    instance.tolerance
    largura = img.shape[1]
    altura = img.shape[0]

    xCentro = int(largura/2)
    yCentro = int(altura/2)

    img = cross(img,xCentro,yCentro)

    # mira central
    line(img, xCentro - instance.tolerance, yCentro - 10, xCentro - instance.tolerance, yCentro + 10)
    line(img, xCentro - instance.tolerance, yCentro - 10, xCentro - instance.tolerance + 10, yCentro - 10 - 10)
    line(img, xCentro - instance.tolerance, yCentro + 10, xCentro - instance.tolerance + 10, yCentro + 10 + 10)
    line(img, xCentro + instance.tolerance, yCentro - 10, xCentro + instance.tolerance, yCentro + 10)
    line(img, xCentro + instance.tolerance, yCentro - 10, xCentro + instance.tolerance - 10, yCentro - 10 - 10)
    line(img, xCentro + instance.tolerance, yCentro + 10, xCentro + instance.tolerance - 10, yCentro + 10 + 10)

    # superior esquerdo
    line(img, instance.tolerance, instance.tolerance, instance.tolerance + 100, instance.tolerance)
    line(img, instance.tolerance, instance.tolerance, instance.tolerance, instance.tolerance + 100)
    # inferior esquerdo
    line(img, instance.tolerance, altura - instance.tolerance, instance.tolerance + 100, altura - instance.tolerance)
    line(img, instance.tolerance, altura - instance.tolerance, instance.tolerance, altura - instance.tolerance - 100)
    # superior direito
    line(img, largura - instance.tolerance, instance.tolerance, largura - instance.tolerance - 100, instance.tolerance)
    line(img, largura - instance.tolerance, instance.tolerance, largura - instance.tolerance, instance.tolerance + 100)
    #inferior direito
    line(img, largura - instance.tolerance, altura - instance.tolerance, largura - instance.tolerance - 100, altura - instance.tolerance)
    line(img, largura - instance.tolerance, altura - instance.tolerance, largura - instance.tolerance, altura - instance.tolerance - 100)

    config = np.array([
        ["Terminator Tasks", [element for element in instance.task if instance.task[element]]],
        ["Terminator VH", instance.visionHeight],
        ["Terminator VW", instance.visionWidth],
        ["Target Color", instance.corCreeper],
        ["Terminator Tolerance",instance.tolerance],
        ["Target Area", instance.area],
        ["Min. Distance", instance.distancia],
        ["Alignment",instance.alignment]

    ])
    counters = np.array([
        ["terminator.counter", instance.counter],
        ["terminator.counterCreeper", instance.counterCreeper],
        ["terminator.counterPista", instance.counterPista],
        ["terminator.counterEstacao", instance.counterEstacao],
        ["terminator.counterLimit", instance.counterLimit]
    ])
    img = textLines(img, config, instance.tolerance, "topLeft")
    img = textLines(img, counters, instance.tolerance, "bottomLeft", altura=instance.visionHeight)
    

    return img
