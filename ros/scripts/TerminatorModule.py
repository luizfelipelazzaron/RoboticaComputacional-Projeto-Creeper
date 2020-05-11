#! /usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function, division

__author__ = ["Cicero Tiago Carneiro Valentim",
              "Luiz Felipe Lazzaron",
              "Thalia Loiola Silva"]

import rospy
import numpy as np
import numpy
import tf
import math
import cv2
import time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
import math
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import visao_module
from auxiliar import *

width = "screen width"
height = "screen height"
tolerance = 0.01

direction = {
    "right": Vector3(0, 0, 1),
    "left": Vector3(0, 0, -1)
}


class Terminator():
    def __init__(self):
        # todos os atributos podem se autoconstruir a
        # partir de valores default
        self.results = []
        self.velocidadeSaida = None
        self.target = None
        self.tolerance = 25
        self.estacao = None
        self.image = None
        # retirados de base_proj.py
        self.cvImage = None
        self.media = []
        self.centro = []
        self.atraso = 1.5E9
        self.area = 0.0
        self.checkDelay = False
        self.x = 0
        self.y = 0
        self.z = 0
        self.id = 0
        self.frame = "camera_link"
        self.tfl = 0
        self.tfBuffer = tf2_ros.Buffer()
        self.bridge = CvBridge()

        self.task = {'iniciar': True,               # o robô inicia sua atividade
                     'procurarPista': False,        # o robô procura visualmente a pista
                     'alcancarPista': False,        # o robô vai em direção a pista
                     'percorrerPista': False,       # o robô percorre a pista
                     'procurarCreeper': False,      # o robô procura visualmente o creeper
                     'alcancarCreeper': False,      # o robô vai em direção ao creeper
                     'pegarCreeper': False,         # a garra pega o creeper
                     'carregarCreeper': False,      # a garra retorna a posição inicial com o creeper
                     'procurarEstacao': False,      # o robô procura visualmente a estação
                     'alcancarEstacao': False,      # o robô vai em direção a estação
                     'soltarCreeper': False,        # a garra solta o creeper
                     'encerrar': False              # o robô encerra sua atividade
                     }

    def acordar(self, imagem):
        self.image = imagem

    def estadoAtual(self):
        if self.task['iniciar']:
            self.iniciar()
        if self.task['procurarPista']:
            self.procurarPista()
        elif self.task['alcancarPista']:
            self.procurarPista()
        elif self.task['percorrerPista']:
            self.percorrerPista(image)
        if self.task['procurarCreeper']:
            self.procurarCreeper()
        elif self.task['alcancarCreeper']:
            self.alcancarCreeper()
        elif self.task['pegarCreeper']:
            self.pegarCreeper()
        elif self.task['carregarCreeper']:
            self.carregarCreeper()
        elif self.task['soltarCreeper']:
            self.soltarCreeper()
        elif self.task['procurarEstacao']:
            self.procurarEstacao()
        elif self.task['alcancarEstacao']:
            self.alcancarEstacao()

        else:
            pass

    def iniciar(self):
        try:
            self.task['iniciar'] = False
            self.task['percorrerPista'] = True
        except:
            pass

    def procurarPista(self):
        pass

    def alcancarPista(self):
        pass

    def percorrerPista(self, image):
        while(True):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                      # Conversão para escala de cinza
            gray = cv2.GaussianBlur(gray, (5, 5), 0.33)                         # Redução de ruído 1# Redução de ruído 1
            limiar, img_limiar = cv2.threshold(                                 # limiarização da imagem ( fonte: https://youtu.be/P2R7Nn1_VwQ )
                gray, 180, 255, cv2.THRESH_BINARY)
            KERNEL = np.array([                                                 # Redução de ruido 2
                [0, 0, 1, 0, 0],                                                # Aplicação de um kernel personalizado ( fonte: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype="uint8")
            segmentado = cv2.morphologyEx(img_limiar, cv2.MORPH_OPEN, KERNEL)
            segmentado = cv2.morphologyEx(segmentado, cv2.MORPH_OPEN, KERNEL)
            segmentado = cv2.morphologyEx(segmentado, cv2.MORPH_OPEN, KERNEL)
            segmentado = cv2.morphologyEx(segmentado, cv2.MORPH_OPEN, KERNEL)
            segmentado = cv2.morphologyEx(segmentado, cv2.MORPH_CLOSE, KERNEL)
            segmentado = cv2.morphologyEx(segmentado, cv2.MORPH_CLOSE, KERNEL)
            segmentado = cv2.morphologyEx(segmentado, cv2.MORPH_CLOSE, KERNEL)
            segmentado = cv2.morphologyEx(segmentado, cv2.MORPH_CLOSE, KERNEL)
            blur = cv2.GaussianBlur(segmentado, (11, 11), 0.33)                 # Redução de ruído 3
            shapes = cv2.Canny(blur, 50, 200)                                   # Detecção de contornos
            lines = cv2.HoughLinesP(                                            # A partir dos contornos, gera um array com linhas que satisfazem os argumentos
                shapes, 1, pi/180, 50, minLineLength=100, maxLineGap=20)        # minLineLength e maxLineGap (parâmetros foram modificados até que um resultado satisfatório fosse atingido)
            print(lines[0])
            OUTPUT = image
            red_lines = []
            blue_lines = []
            try:
                m1, n1 = coeficientes(red_lines)                                # função 'coeficientes' definida lá em cima
                m2, n2 = coeficientes(blue_lines)                               # m e n são conficientes de retas da forma:  y = mx + n
            except:
                pass
            finally:
                X = max(0, min(int((n2-n1)/(m1-m2)), WIDTH))                    # min() e max() servem para deixar o circulo sempre visível
                Y = max(0, min(int((m1*X + n1)), HEIGHT))                       # X e Y são as coordenadas do ponto de encontro entre a reta média azul e a reta média vermelha
                cv2.circle(OUTPUT, (X, Y), 10, (0, 255, 0), 2, 2)               # print("(X,Y) =",(X,Y)) # descomente essa linha para printar no terminal as coordenadas do centro
            cv2.putText(OUTPUT," Aperte q para sair", (0,50), font, 1,(255,255,255),2,cv2.LINE_AA) # Adicionar textos na tela
            cv2.imshow("output", OUTPUT)                                        # Exibir o resultado
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # try:
        #     self.task['percorrerPista'] = False
        #     self.task['procurarCreeper'] = True
        # except:
        #     pass

    def procurarCreeper(self):

        pass

    def alcancarCreeper(self):
        pass

    def pegarCreeper(self):
        pass

    def carregarCreeper(self):
        pass

    def procurarEstacao(self):
        pass

    def alcancarEstacao(self):
        pass

    def soltarCreeper(self):
        pass

    def procurarEstacao(self):
        if self.targetInCenter([result[2], result[3]]):
            print("target centralized:", result[0])
            self.target = "cat"
            self.stop()
        else:
            self.move(0, -0.1)
            self.target = None
            if self.target == "cat":
                self.move(1, 0)
            else:
                pass

    def targetInCenter(self, targetPosition):
        # targetPosition é da forma [(90, 141), (177, 265)]
        # targetPosition[0] = canto superior esquerdo
        # targetPosition[1] = canto inferior direito
        xTargetCenter = (result[2][0] + result[3][0])/2
        xTerminatorCenter = self.centro[0]
        return abs(xTargetCenter-xTerminatorCenter) <= self.tolerance

    def move(self, velocidadeTangencial, velocidadeAngular):
        velocidade = Twist(Vector3(velocidadeTangencial, 0, 0),
                           Vector3(0, 0, velocidadeAngular))
        self.velocidadeSaida.publish(velocidade)
        rospy.sleep(0.1)

    def stop(self):
        self.move(velocidadeTangencial=0, velocidadeAngular=0)

    def recebe(self, msg):
        for marker in msg.markers:
            self.id = marker.id
            marcador = "ar_marker_" + str(self.id)

            print(self.tfBuffer.can_transform(
                self.frame, marcador, rospy.Time(0)))

            # Terminamos
            print("id: {}".format(self.id))

    def identificaEstacao(self, imagem):
        # print("frame")
        # global terminator.media
        now = rospy.get_rostime()
        imgtime = imagem.header.stamp
        lag = now-imgtime  # calcula o lag
        delay = lag.nsecs
        # print("delay ", "{:.3f}".format(delay/1.0E9))
        if delay > self.atraso and self.checkDelay == True:
            print("Descartando por causa do delay do frame:", delay)
            return
        try:
            antes = time.clock()
            self.cvImage = self.bridge.compressed_imgmsg_to_cv2(
                imagem, "bgr8")
            # Note que os resultados já são guardados automaticamente na variável
            # chamada resultados
            self.centro, self.imagem, self.resultados = visao_module.processa(
                self.cvImage)
            for resultado in self.resultados:
                estacao = resultado[2]
                print(estacao)
                pass

            depois = time.clock()
            # Desnecessário - Hough e MobileNet já abrem janelas
            # cv2.imshow("Camera", cvImage)
        except CvBridgeError as e:
            print('ex', e)
