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
from sensor_msgs.msg import Image, LaserScan
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
import rospkg
import os
# from garra_demo import MoveGroupPythonIntefaceTutorial, all_close

width = "screen width"
height = "screen height"
tolerance = 0.01

direction = {
    "right": Vector3(0, 0, 1),
    "left": Vector3(0, 0, -1)
}

#Instanciamento da Rede Neural
rospack = rospkg.RosPack()
path = rospack.get_path('ros')
scripts = os.path.join(path, "scripts")
proto = os.path.join(scripts,"MobileNetSSD_deploy.prototxt.txt")
model = os.path.join(scripts, "MobileNetSSD_deploy.caffemodel")
net = cv2.dnn.readNetFromCaffe(proto,model) 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	        "sofa", "train", "tvmonitor"]
CONFIDENCE = 0.7
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

verde = [1, 255, 7]
azul = [20, 145, 253]
rosa = [255, 0, 252]

class Terminator():
    def __init__(self):
        # todos os atributos podem se autoconstruir a
        # partir de valores default
        self.c = 0
        self.corCreeper = verde
        self.estacaoEscolhida = 'dog'
        self.results = []
        self.velocidadeSaida = None
        self.target = None
        self.tolerance = 20
        self.estacao = None
        self.image = None
        self.counter = 0
        self.counterCreeper = 0
        self.counterPista = 0
        self.counterEstacao = 0
        self.counterLimit = 7
        self.distancia = 30         #valor grande , isto é, maior que 0.2
        self.yFindOutSpeedway = 400
        self.rotationMode = False
        self.dataScan = None
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
        self.finalImage = None
        
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
            if self.task['alcancarPista']:
                self.procurarPista()
            if self.task['percorrerPista']:
                self.percorrerPista()
            # Tasks relacionados ao Creeper
            if self.task['procurarCreeper']:
                self.procurarCreeper()          
            if self.task['alcancarCreeper']:
                self.alcancarCreeper()
            if self.task['pegarCreeper']:
                self.pegarCreeper()
            if self.task['carregarCreeper']:
                self.carregarCreeper()
            if self.task['soltarCreeper']:
                self.soltarCreeper()
            # Tasks relacionados à Estação
            if self.task['procurarEstacao']:
                self.procurarEstacao()
            if self.task['alcancarEstacao']:
                self.alcancarEstacao()
            # self.imShow()
        else:
            pass

    def iniciar(self):
        self.task['iniciar'] = False
        self.task['procurarCreeper'] = True
        self.task['percorrerPista'] = True
        print("Mudando de Estado | tasks ativas:")
        [print(element) for element in self.task if self.task[element]]


    def procurarPista(self):
        # print("Ativando modo de procurar a Pista")
        colorSpeedwayBorder =[255, 255, 0]
        if self.counterPista < self.counterLimit:
            try:
                self.identifica_cor(colorSpeedwayBorder)
                print("y: ",self.media[1])
                print("distancia minima detectada :", self.distancia)
                if self.distancia < 0.6:
                    self.move(-0.2,0)
                    self.media = None
                elif self.media[1] < self.yFindOutSpeedway:
                    if self.targetInCenter(self.media) and not self.rotationMode:
                        self.move(0.2, 0)
                    else:
                        self.move(0.05, 4*self.whereTo(self.media[0]))
                else:
                    self.rotationMode = True
                    if self.targetInCenter(self.media):
                        self.stop()
                        self.counterPista += 1
                        self.rotationMode = False
                        print("self.counter:", self.counter)
                    else:
                        self.move(0.1,4*self.whereTo(self.media[0]))
                        self.counterPista = 0
            except:
                pass
        else:
            self.task['procurarPista'] = False
            self.task['percorrerPista'] = True
            self.counterPista = 0
            print("Mudando de Estado | tasks ativas:")
            [print(element) for element in self.task if self.task[element]]

    def alcancarPista(self):
        pass

    def percorrerPista(self):
        # print("Ativando modo de percorrer a Pista")
        if self.counterPista < self.counterLimit:
            try:
                localTarget = self.followPath()
                if self.targetInCenter(localTarget):
                    print("linha reta")
                    self.move(0.5, 0)
                else:
                    self.move(0.1, self.whereTo(localTarget[0]))
            except:
                self.counterPista += 1
                print("contador do percorrerPista: ", self.counterPista)
        else:
            self.task['percorrerPista'] = False
            self.counterPista = 0
            if self.task['procurarCreeper']:
                pass  
            else:
                self.task['procurarPista'] = True


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
        # print("Ativando modo de prcurar o Creeper")
        if self.counterCreeper < self.counterLimit:
            try:
                self.identifica_cor(self.corCreeper)
                if self.area > 900:
                    print("achou creeper \o/")
                    print("area detectada = ", self.area)
                if self.area > 2500:
                    self.counterCreeper += 1
                    print("Contador: ", self.counterCreeper)
                else:
                    self.counterCreeper = 0
            except:
                pass
        else:
            self.task['percorrerPista'] = False
            self.task['procurarPista'] = False
            self.task['procurarCreeper'] = False
            self.task['alcancarCreeper'] = True
            print("Mudando de Estado | tasks ativas:")
            [print(element) for element in self.task if self.task[element]]
            self.counterCreeper = 0

    def alcancarCreeper(self):
        # print("Ativando modo de alcancar o Creeper")
        if self.counterCreeper < self.counterLimit:
            try:
                self.identifica_cor(self.corCreeper)
                print("area detectada = ", self.area)
                if self.area > 11000:
                    self.counterCreeper += 1
                else:
                    self.counterCreeper = 0
                if self.media[1] < self.yFindOutSpeedway: 
                    if self.targetInCenter(self.media) and not self.rotationMode:
                        self.move(0.2, 0)
                    else:
                        self.move(0.1, self.whereTo(self.media[0]))
                else:
                    self.rotationMode = True
                    if self.targetInCenter(self.media):
                        self.stop()
                        self.counterCreeper += 1
                    else:
                        self.move(0.0, self.whereTo(self.media[0]))
                        self.counterCreeper = 0
            except:
                pass
        else:
            self.stop()
            self.task['alcancarCreeper'] = False
            self.task['procurarPista'] = True
            self.media = None
            print("Mudando de Estado | tasks ativas:")
            [print(element) for element in self.task if self.task[element]]



    def pegarCreeper(self):
        print("Bora aprender a pegar o Creeper então")
        input("escreva None")
        self.task['pegarCreeper'] = False
        self.task['procurarEstacao'] = True
        # garra = MoveGroupPythonIntefaceTutorial()
        # garra.open_gripper()

    def carregarCreeper(self):
        pass

    def procurarEstacao(self):
        if self.counter < self.counterLimit:
            try:
                self.identificaObjetos()
                estacaoEncontrada = self.results[0][0]
                if estacaoEncontrada == self.estacaoEscolhida:
                    self.counter += 1
                    print("Contador: ", self.counter)
                else:
                    self.counter = 0
            except:
                pass
        else:
            self.counter = 0
            self.task['procurarEstacao'] = False
            print("Mudando de Estado | tasks ativas:")
            [print(element) for element in self.task if self.task[element]]

    def alcancarEstacao(self):
        pass

    def soltarCreeper(self):
        pass

    # def procurarEstacao(self):
    #     if self.targetInCenter([result[2], result[3]]):
    #         print("target centralized:", result[0])
    #         self.target = "cat"
    #         self.stop()
    #     else:
    #         self.move(0, -0.1)
    #         self.target = None
    #         if self.target == "cat":
    #             self.move(1, 0)
    #         else:
    #             pass

    def imShow(self):
        if self.finalImage is not None:
            thisImage = self.finalImage
            thisImage = aux.drawHUD(thisImage, self.tolerance)
            cv2.imshow("Final Image", thisImage)
            cv2.waitKey(1)

    def targetInCenter(self, targetPosition):
        """targetPosition é da forma (x,y)"""
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
            # self.finalImage = self.cvImage
            if not (self.visionHeight and self.visionWidth):
                print(self.cvImage.shape)
                self.visionWidth = self.cvImage.shape[1]
                self.visionHeight = self.cvImage.shape[0]
                print("(Terminator.visionWidth, Terminator.visionHeight): ({0},{1})".format(
                    self.visionWidth, self.visionHeight))

            
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
            for leftLine in left:
                aux.draw_line(frame, leftLine, color=(32, 0, 255))
        except:
            self.finalImage = frame
            return 0, (self.visionHeight/2)
        try:
            right = np.array(right)
            for rightLine in right:
                aux.draw_line(frame, rightLine, color=(255, 32, 0))
        except:
            self.finalImage = frame
            return int(self.visionWidth), (self.visionHeight/2)

        try:
            # reta média esquerda
            m1, n1 = aux.coefficients(left)
            # print("m1,n1:", m1, n1)
        except:
            self.finalImage = frame
            return 0, int(self.visionHeight/2)
        try:
            # reta média direita
            m2, n2 = aux.coefficients(right)
            # print("m2,n2", m2, n2)
        except:
            self.finalImage = frame
            return int(self.visionWidth), int(self.visionHeight/2)
        # min() e max() servem para deixar o circulo sempre visível
        # X e Y são as coordenadas do ponto de encontro entre a reta média azul e a reta média vermelha
        X = max(0, min(int((n2-n1)/(m1-m2)), self.visionWidth))

        # Y = max(0, min(int((m1*X + n1)), self.visionHeight))
        Y = int(self.visionHeight/2)
        # print("(X,Y) =",(X,Y)) # descomente essa linha para printar no terminal
        # as coordenadas do centro
        cv2.circle(frame, (X, Y), 10, (0, 255, 0), 2, 2)
        self.finalImage = frame
        return X, Y

    def pathFinder(self):
        """Encontra a pista se não estiver nela, se dirige até o
         centro e alinha com a faixa pontilhada central (eu espero);"""
        pass

    def identifica_cor(self, colorRgb):
        '''
        Segmenta o maior objeto cuja cor é parecida com cor_h (HUE da cor, no espaço HSV).
        '''

        # No OpenCV, o canal H vai de 0 até 179, logo cores similares ao
        # vermelho puro (H=0) estão entre H=-8 e H=8.
        # Precisamos dividir o inRange em duas partes para fazer a detecção
        # do vermelho:
        if not self.task['percorrerPista']:
            self.finalImage = self.cvImage
        frame = self.cvImage
        frame_hsv = cv2.cvtColor(self.cvImage, cv2.COLOR_BGR2HSV)

        cor_menor,cor_maior = aux.ranges(colorRgb) # devolve dois valores: hsv_menor e hsv_maior
        segmentado_cor = cv2.inRange(frame_hsv, cor_menor, cor_maior)
    
        # Note que a notacão do numpy encara as imagens como matriz, portanto o enderecamento é
        # linha, coluna ou (y,x)
        # Por isso na hora de montar a tupla com o centro precisamos inverter, porque
        centro = (int(self.visionWidth/2), int(self.visionHeight/2))

        # A operação MORPH_CLOSE fecha todos os buracos na máscara menores
        # que um quadrado 7x7. É muito útil para juntar vários
        # pequenos contornos muito próximos em um só.
        segmentado_cor = cv2.morphologyEx(segmentado_cor,cv2.MORPH_CLOSE,np.ones((7, 7)))

        # Encontramos os contornos na máscara e selecionamos o de maior área
        #contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maior_contorno = None
        maior_contorno_area = 0

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > maior_contorno_area:
                maior_contorno = cnt
                maior_contorno_area = area

        # Encontramos o centro do contorno fazendo a média de todos seus pontos.
        if not maior_contorno is None and maior_contorno_area > 500:
            cv2.drawContours(self.finalImage, [maior_contorno], -1, [0, 0, 255], 5)
            maior_contorno = np.reshape(maior_contorno, (maior_contorno.shape[0], 2))
            media = maior_contorno.mean(axis=0)
            media = media.astype(np.int32)
            cv2.circle(self.finalImage, (media[0], media[1]), 5, [0, 255, 0])
            aux.cross(self.finalImage, centro[0], centro[1])
        else:
            media = (0, 0)

        # Representa a area e o centro do maior contorno no frame
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame,"{:d} {:d}".format(*media),(20,100), 1, 4,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame,"{:0.1f}".format(maior_contorno_area),(20,50), 1, 4,(255,255,255),2,cv2.LINE_AA)
        # self.maior_contorno_area = maior_contorno_area
        self.media = media
        self.centro = centro
        self.area = area
        # self.finalImage = frame

    def scanTarget(self, dataScan):
        self.distancia = np.array(dataScan.ranges).round(decimals=2)[0]

    def identificaObjetos(self):
        imagem = self.cvImage
        blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 0.007843, (300, 300), 127.5)
        # print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        (w, h) = (self.visionWidth,self.visionHeight)
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                # print("[INFO] {}".format(label))
                cv2.rectangle(imagem, (startX, startY),(endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(imagem, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,COLORS[idx], 2)
                self.results.append((CLASSES[idx], confidence *100, (startX, startY), (endX, endY)))
        