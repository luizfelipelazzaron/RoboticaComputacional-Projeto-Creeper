#! /usr/bin/env python
# -*- coding:utf-8 -*-
from libraries import *
import cormodule
import auxiliar
        
class Creeper():
    def __init__(self,identification,color,image):
        self.identification = identification
        self.color = color
        #As especificações a seguir extraídas 
        # do projeto 4 de cor.py
        #Todos os atributos podem se autoconstruir
        #a partir de valores default
        self.distance = None
        self.bridge = CvBridge()
        self.cv_image = None
        self.average = []
        self.center = []
        self.atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos
        self.area = 0.0 # Variavel com a area do maior contorno
        self.check_delay = False
        self.image = image

    def find_creeper(self,image):
        global cv_image
        global average
        global center 
        now = rospy.get_rostime()
        imgtime = image.header.stamp
        lag = now-imgtime # calcula o lag
        delay = lag.nsecs
        print("delay ", "{:.3f}".format(delay/1.0E9))
        pass

    def go_to_creeper(self):
        pass

    def play(self):
        pass

