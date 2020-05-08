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


cv_image = None
atraso = 1.5E9  # 1 segundo e meio. Em nanossegundos



# A função a seguir é chamada sempre que chega um novo frame


if __name__ == "__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/rgb/image_raw/compressed"

    t800 = Terminator()

    recebedorEstacao = rospy.Subscriber(
        topico_imagem, CompressedImage, t800.estadoAtual, queue_size=4, buff_size=2**24)
    # Para recebermos notificacoes de que marcadores foram vistos
    
    recebedorId = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, t800.recebe)

    print("Usando ", topico_imagem)

    t800.velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    tolerancia = 25
    # Exemplo de categoria de resultados
    # [('chair', 86.965459585189819, (90, 141), (177, 265))]

    try:
        # Inicializando - por default gira no sentido anti-horário
        # vel = Twist(Vector3(0,0,0), Vector3(0,0,math.pi/10.0))

        while not rospy.is_shutdown():
            t800.dealWithResults()
            

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")
