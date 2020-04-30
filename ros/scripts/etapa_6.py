#! /usr/bin/env python
# -*- coding:utf-8 -*-


import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

v = 0.1  # Velocidade linear

distancia = None #Definindo distância

def scaneou(dado):
    global distancia #Definindo distância como uma variável global
    # print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
    # print("Leituras:")
    # print(np.array(dado.ranges).round(decimals=2))
    distancia = np.array(dado.ranges).round(decimals=2)[0]
    # print("Intensities")
    # print(np.array(dado.intensities).round(decimals=2))9  2.69

if __name__=="__main__":
    rospy.init_node("etapa_6")

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

    # if min(np.array(dado.ranges).round(decimals=2)) < 1.02:
    #     v = -0.1
    # else:
    #     v = 0.1

    try:
        while not rospy.is_shutdown():
            print(distancia)
            if distancia < 1.02:
                velocidade = Twist(Vector3(-0.1, 0, 0), Vector3(0, 0, 0))
                velocidade_saida.publish(velocidade)
                rospy.sleep(2)
            else:
                velocidade = Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0))
                velocidade_saida.publish(velocidade)
                rospy.sleep(2)
                    
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")
