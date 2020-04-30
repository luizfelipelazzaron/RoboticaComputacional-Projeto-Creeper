#! /usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function, division

__author__ = ["Cicero Tiago Carneiro Valentim",
              "Luiz Felipe Lazzaron", "Thalia Loiola Silva"]

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

width = "screen width"
height = "screen height"
tolerancia = 0.01

direction = {
    "right": Vector3(0, 0, 1),
    "left": Vector3(0, 0, -1)
}


class Terminator():
    def __init__(self):
        # todos os atributos podem se autoconstruir a
        # partir de valores default

        self.resultados = []
        self.velocidade_saida = None

        # retirados de base_proj.py
        self.cv_image = None
        self.media = []
        self.centro = []
        self.atraso = 1.5E9 
        self.area = 0.0
        self.check_delay = False
        self.x = 0
        self.y = 0
        self.z = 0
        self.id = 0
        self.frame = "camera_link"
        self.tfl = 0
        self.tf_buffer = tf2_ros.Buffer()

    def move(self):
        vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.1))
        self.velocidade_saida.publish(vel)

    def recebe(self, msg):

        for marker in msg.markers:
            self.id = marker.id
            marcador = "ar_marker_" + str(self.id)

            print(tf_buffer.can_transform(self.frame, marcador, rospy.Time(0)))
            header = Header(frame_id=marcador)
            # Procura a transformacao em sistema de coordenadas entre a base do robo e o marcador numero 100
            # Note que para seu projeto 1 voce nao vai precisar de nada que tem abaixo, a
            # Nao ser que queira levar angulos em conta
            trans = tf_buffer.lookup_transform(frame, marcador, rospy.Time(0))

            # Separa as translacoes das rotacoes
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            # ATENCAO: tudo o que vem a seguir e'  so para calcular um angulo
            # Para medirmos o angulo entre marcador e robo vamos projetar o eixo Z do marcador (perpendicular)
            # no eixo X do robo (que e'  a direcao para a frente)
            t = transformations.translation_matrix([x, y, z])
            # Encontra as rotacoes e cria uma matriz de rotacao a partir dos quaternions
            r = transformations.quaternion_matrix(
                [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            # Criamos a matriz composta por translacoes e rotacoes
            m = numpy.dot(r, t)
            # Sao 4 coordenadas porque e'  um vetor em coordenadas homogeneas
            z_marker = [0, 0, 1, 0]
            v2 = numpy.dot(m, z_marker)
            v2_n = v2[0:-1]  # Descartamos a ultima posicao
            n2 = v2_n/linalg.norm(v2_n)  # Normalizamos o vetor
            x_robo = [1, 0, 0]
            # Projecao do vetor normal ao marcador no x do robo
            cosa = numpy.dot(n2, x_robo)
            angulo_marcador_robo = math.degrees(math.acos(cosa))

            # Terminamos
            print("id: {}".format(id))



