import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#%matplotlib inline

#1. Printando a Versão do OpenCV
print(cv2.__version__)

#2
print(os.getcwd())

#3
proto = "./mobilenet_detection/MobileNetSSD_deploy.prototxt.txt"
model = "./mobilenet_detection/MobileNetSSD_deploy.caffemodel"

#4 Intanciação da rede neural
net = cv2.dnn.readNetFromCaffe(proto, model)

#5 Categorias da MobileNet
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
#6 Lendo a Imagem
#cap = cv2.VideoCapture("VID_20200302_063445951.mp4")
cap = cv2.VideoCapture(0)
CONFIDENCE = 0.7
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#7 Função Detect
def detect(frame):
    """
        Recebe - uma imagem colorida
        Devolve: objeto encontrado
    """
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results

#10 Vídeo
while(True):

# Lendo a imagem
    ret,frame = cap.read()

# Colocando a imagem na função detect
    saida, resultados = detect(frame)

# Printando os resultados no consolse
    print(resultados)

# Plotando a imagem na tela
    cv2.imshow("Saida", saida)

# Atalho para sair q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()