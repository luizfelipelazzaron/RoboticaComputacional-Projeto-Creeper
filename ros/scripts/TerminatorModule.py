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
import auxiliar as aux
import visao_module


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
        self.tolerance = 20
        self.estacao = None
        self.image = None
        self.counter = 0
        self.counterLimit = 5
        # retirados de base_proj.py
        self.cvImage = None
        self.visionHeight = None
        self.visionWidth = None
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
        if self.cvImage is not None:
            if self.task['iniciar']:
                self.iniciar()
            # Tasks relacionados a Pista
            if self.task['procurarPista']:
                self.procurarPista()
            elif self.task['alcancarPista']:
                self.procurarPista()
            elif self.task['percorrerPista']:
                self.percorrerPista()
            # Tasks relacionados ao Creeper
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
            # Tasks relacionados à Estação
            elif self.task['procurarEstacao']:
                self.procurarEstacao()
            elif self.task['alcancarEstacao']:
                self.alcancarEstacao()
        else:
            pass

    def iniciar(self):
        self.task['iniciar'] = False
        self.task['percorrerPista'] = True
        self.task['procurarCreeper'] = True

    def procurarPista(self):
        if self.counter < self.counterLimit:
            frame = self.cvImage
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cor_menor,cor_maior = aux.ranges([62,100,100]) # amarelo
            centro = (frame.shape[1]//2, frame.shape[0]//2)
        else:
            print("mudando de estado")
            self.task['procurarPista'] = False
            self.task['percorrerPista'] = True
            self.counter = 0 

    def alcancarPista(self):
        pass

<<<<<<< HEAD
    def percorrerPista(self):
        if self.counter < self.counterLimit:
            try:
                localTarget = self.followPath()

                if self.targetInCenter(localTarget):
                    print("linha reta")
                    self.move(0.5, 0)
                else:
                    # self.stop()
                    self.move(0.08, self.whereTo(localTarget[0]))
                cv2.circle(self.cvImage, (localTarget[0], localTarget[1]), 10, (0, 255, 0), 2, 2)
                cv2.imshow("Terminator Vision", self.cvImage)
                cv2.waitKey(1)
            except:
                self.counter += 1
                print("contador: ", self.counter)
=======
       def percorrerPista(self):
        localTarget = self.followPath()

        if self.targetInCenter(localTarget):
            print("linha reta")
            self.move(0.5, 0)
>>>>>>> 7935eaa77bb6dadb6618944af50c7142a8596305
        else:
            print("Deu certo")
            self.task['percorrerPista'] = False
            self.task['procurarPista'] = True
            self.counter = 0
 
    def whereTo(self, x):
        if x > self.visionWidth/2:
            # corrigir para a esquerda
            print("corrigir para a direita")
            return -0.05
        elif x < self.visionWidth/2:
            print("corrigir para a esquerda")
            # corrigir para a direita
            return 0.05

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
        # targetPosition é da forma (x,y)
        # targetPosition[0] = canto superior esquerdo
        # targetPosition[1] = canto inferior direito
        xTargetCenter = targetPosition[0]
        xTerminatorCenter = self.visionWidth/2
        return abs(xTargetCenter-xTerminatorCenter) <= self.tolerance

    def move(self, velocidadeTangencial, velocidadeAngular):
        """Bom valor para velocidade tangencial: 0.5\n
        Bom valor para velocidade angular: 0.1"""
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
        pass

    def processImage(self, image):
        """Responsável por receber a imagem da câmera do robô;\n
        Ela faz o processamento inicial;"""
        now = rospy.get_rostime()
        imgtime = image.header.stamp
        lag = now-imgtime  # calcula o lag
        delay = lag.nsecs
        # print("delay ", "{:.3f}".format(delay/1.0E9))
        if delay > self.atraso and self.checkDelay == True:
            print("Descartando por causa do delay do frame:", delay)
            return
        try:
            antes = time.clock()
            # obter imagem a ser manipulada pelo openCV;
            # ela foi armazenada em um atributo para que
            # possa ser facilemente manipulada por outros métodos;
            self.cvImage = self.bridge.compressed_imgmsg_to_cv2(image, "bgr8")
            if not (self.visionHeight and self.visionWidth):
                print(self.cvImage.shape)
                self.visionWidth = self.cvImage.shape[1]
                self.visionHeight = self.cvImage.shape[0]
                print("(Terminator.visionWidth, Terminator.visionHeight): ({0},{1})".format(
                    self.visionWidth, self.visionHeight))
                
            # aux.cross(self.cvImage, self.visionWidth/2, self.visionHeight/2)
            aux.drawHUD(self.cvImage, self.visionWidth/2, self.visionHeight/2, self.tolerance)
            depois = time.clock()

        except CvBridgeError as e:
            print('ex', e)

    def followPath(self):
        """
        Manipulação necessária e suficiente para: `seguir a pista`;\n
        ATENÇÃO: esse método não localiza a pista se estiver fora dela.\n
        Se quiser localizar a pista, use `pathFinder`; 
        """
        # Vamos chamar de frame só pra manter o costume;
        frame = self.cvImage
        # converter para cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # blur para deixar lisinho
        # nunca entendi o que é o 0.33
        gray = cv2.GaussianBlur(gray, (5, 5), 0.33)

        # limiarização da imagem: filtrar o que for claro (branco)
        threshold = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

        # kernel customizado para a nossa a pista:
        KERNEL = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype="uint8")

        # segmentar para modelar as bordas:
        segmented = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel=KERNEL)
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel=KERNEL)

        # outro blur
        # threshold = cv2.GaussianBlur(threshold, (5, 5), 0.33)

        # detectar formas
        shapes = cv2.Canny(segmented, 50, 200)

        # encontrar linas
        lines = cv2.HoughLinesP(shapes, 1, np.pi/180,
                                50, minLineLength=100, maxLineGap=20)

        # linha da esquerda != linha da direita
        if lines is not None:

            try:
                left = list(filter(lambda line: -1.3 <
                                   aux.tangente(line) < -0.2, lines))
                right = list(filter(lambda line: 0.3 <
                                    aux.tangente(line) < 19, lines))
            except:
                print("caiu no except")
                pass
        # agora que temos listas de linhas com inclinaçao
        # negativa (esquerda) e positiva (direita), transformamos em array
        # para executar umas operações mais sucintas
        try:
            left = np.array(left)
        except:
            return 0, (self.visionHeight/2)
        try:
            right = np.array(right)
        except:
            return int(self.visionWidth), (self.visionHeight/2)
        # print("left:", left.shape)
        # print("right:", right.shape)
        # for leftLine in left:
        #     aux.draw_line(self.cvImage, leftLine, color=(32, 0, 255))
        # for rightLine in right:
        #     aux.draw_line(self.cvImage, rightLine, color=(255, 32, 0))

        try:
            # reta média esquerda
            m1, n1 = aux.coefficients(left)
            # print("m1,n1:", m1, n1)
        except:
            return 0, int(self.visionHeight/2)
        try:
            # reta média direita
            m2, n2 = aux.coefficients(right)
            # print("m2,n2", m2, n2)
        except:
            return int(self.visionWidth), int(self.visionHeight/2)
        # min() e max() servem para deixar o circulo sempre visível
        # X e Y são as coordenadas do ponto de encontro entre a reta média azul e a reta média vermelha
        X = max(0, min(int((n2-n1)/(m1-m2)), self.visionWidth))

        # Y = max(0, min(int((m1*X + n1)), self.visionHeight))
        Y = int(self.visionHeight/2)
        # print("(X,Y) =",(X,Y)) # descomente essa linha para printar no terminal
        # as coordenadas do centro
        

        return X, Y

    def pathFinder(self):
        """Encontra a pista se não estiver nela, se dirige até o
         centro e alinha com a faixa pontilhada central (eu espero);"""
        pass
