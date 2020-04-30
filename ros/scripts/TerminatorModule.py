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

import visao_module


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

        self.results = []
        self.velocidade_saida = None
        self.target = {}
        self.tolerancia = 25


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
        self.bridge = CvBridge()

    def move(self, velocidade_tangencial, velocidade_angular):
        velocidade = Twist( Vector3(velocidade_tangencial, 0, 0), Vector3(0, 0, velocidade_angular))
        self.velocidade_saida.publish(velocidade)
        rospy.sleep(0.1)

    def stop(self):
        self.move(velocidade_tangencial=0,velocidade_angular=0)

    def dealWithResults(self):
        if not self.target:
            for result in self.results:
                if result[0] == "cat" and self.targetInCenter( [result[2],result[3]] ):
                    # print("self.velocidade_saida:",self.velocidade_saida)
                    # print(self.centro)
                    print("target detected:",result[0])
                    self.target = "cat"
                    self.stop()
                else:
                    self.move(0,-0.1)
                    self.target = None

        elif self.target == "cat":
            self.move(1,0)
        else:
            pass

    def targetInCenter(self, target_position):
        # target_position é da forma [(90, 141), (177, 265)]
        # target_position[0] = canto superior esquerdo
        # target_position[1] = canto inferior direito
        

        x_do_centro_do_target =  (result[2][0] + result[3][0])/2
        x_do_centro_do_terminator = self.centro[0]

        return abs(x_do_centro_do_target-x_do_centro_do_terminator) <= self.tolerancia


    def recebe(self, msg):
        for marker in msg.markers:
            self.id = marker.id
            marcador = "ar_marker_" + str(self.id)

            print(self.tf_buffer.can_transform(
                self.frame, marcador, rospy.Time(0)))

            # Terminamos
            print("id: {}".format(self.id))

    def processaFrame(self, imagem):
        # print("frame")
        # global terminator.media

        now = rospy.get_rostime()
        imgtime = imagem.header.stamp
        lag = now-imgtime  # calcula o lag
        delay = lag.nsecs
        # print("delay ", "{:.3f}".format(delay/1.0E9))
        if delay > self.atraso and self.check_delay == True:
            print("Descartando por causa do delay do frame:", delay)
            return
        try:
            antes = time.clock()
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(
                imagem, "bgr8")
            # Note que os resultados já são guardados automaticamente na variável
            # chamada resultados
            self.centro, self.imagem, self.results = visao_module.processa(
                self.cv_image)

            depois = time.clock()
            # Desnecessário - Hough e MobileNet já abrem janelas
            #cv2.imshow("Camera", cv_image)
        except CvBridgeError as e:
            print('ex', e)

