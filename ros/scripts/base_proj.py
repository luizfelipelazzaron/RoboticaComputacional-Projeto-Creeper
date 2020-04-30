#! /usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function, division
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
from TerminatorModule import Terminator

import visao_module


bridge = CvBridge()

cv_image = None
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos


terminator = Terminator()

# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    # global terminator.media
    
    now = rospy.get_rostime()
    imgtime = imagem.header.stamp
    lag = now-imgtime  # calcula o lag
    delay = lag.nsecs
    # print("delay ", "{:.3f}".format(delay/1.0E9))
    if delay > terminator.atraso and terminator.check_delay == True:
        print("Descartando por causa do delay do frame:", delay)
        return
    try:
        antes = time.clock()
        terminator.cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        # Note que os resultados já são guardados automaticamente na variável
        # chamada resultados
        terminator.centro, terminator.imagem, terminator.resultados = visao_module.processa(terminator.cv_image)
        for r in terminator.resultados:
            # print(r) - print feito para documentar e entender
            # o resultado
            pass

        depois = time.clock()
        # Desnecessário - Hough e MobileNet já abrem janelas
        #cv2.imshow("Camera", cv_image)
    except CvBridgeError as e:
        print('ex', e)


if __name__ == "__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/rgb/image_raw/compressed"

    recebedor = rospy.Subscriber(
        topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size=2**24)
    # Para recebermos notificacoes de que marcadores foram vistos
    recebedor = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, terminator.recebe)

    print("Usando ", topico_imagem)

    terminator.velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    tolerancia = 25

    # Exemplo de categoria de resultados
    # [('chair', 86.965459585189819, (90, 141), (177, 265))]

    try:
        # Inicializando - por default gira no sentido anti-horário
        # vel = Twist(Vector3(0,0,0), Vector3(0,0,math.pi/10.0))

        while not rospy.is_shutdown():
            for r in terminator.resultados:
                print(r)
            # velocidade_saida.publish(vel)
            rospy.sleep(0.1)

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")

