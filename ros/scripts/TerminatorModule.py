#! /usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = ["Cicero Tiago Carneiro Valentim",
              "Luiz Felipe Lazzaron", "Thalia Loiola Silva"]

from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped

width = "screen width"
height = "screen height"
tolerancia = 0.01*width

direction = {
    "right": Vector3(0, 0, 1),  # ????
    # ???? não lembro como é o código de velocidade angular no sentido
    # horário e no sentido anti-horário
    "left": Vector3(0, 0, 2)
}


class Terminator():
    def __init__(self):
        # todos os atributos podem se autoconstruir a
        # partir de valores default
        self.detections = detections
        self.goAhead = False
        self.turnRight = False
        self.turnLeft = False
        self.stop = False
        self.target = None
        self.pista_em_ingles = self.findPistaEmIngles()

    def findPistaEmIngles(self):
        """retorna 'right', 'left' ou 'centered'"""
        # codigo que capitura frame e identifica se a pista está à
        # esquerda, à direita ou suficientemente centralizado

    def moveStraight(self):
        pass  # commando que manda o robô pra frente aqui

    def turn(self, whereTo):  # exemplo: turn(terminato.turnRight)
        pass  # comando que adiciona velocidade angular ao robô
        velocidade_do_robo = direction[whereTo]

    def newDetections(self, output_mobilenet):
        pass  # verifica as detecções e atualiza self.detections
        self.detections = "newDetections_from_output_mobilenet"
        return "newDetections_from_output_mobilenet" not in self.detections

    def findDirections(self):
        # se o alvo for detectado:
        if self.target:
            if self.target.x < width/2 - tolerancia:
                self.turnRight = True
            elif self.target.x > width/2 + tolerancia:
                self.turnLeft = True
        else:
            self.turnLeft = False
            self.turnRight = False
            self.autoDestroy(self)

    def whatToDo(self):
        # relativo à orientação:
        if self.goAhead:
            self.moveStraight()
        elif self.turnRight:
            self.turn("right")
        elif self.turnLeft:
            self.turn("left")

        # relativo a novas detecções
        if self.newDetections("output das detecções"):
            self.stop()
            self.findDirections(self.detections)
        else:
            pass
