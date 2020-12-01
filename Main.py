import cv2
import numpy as np
import os

import DetectarCaracteres
import DetectarPlacas

ESCALA_PRETO = (0.0, 0.0, 0.0)
ESCALA_BRANCO = (255.0, 255.0, 255.0)
ESCALA_AMARELO = (0.0, 255.0, 255.0)
ESCALA_VERDE = (0.0, 255.0, 0.0)
ESCALA_VERMELHO = (0.0, 0.0, 255.0)

mostrarPassos = False

def main():
    blnKNNTrainingSuccessful = DetectarCaracteres.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print ("\nerro: o treinamento KNN não foi bem sucedido\n")
        return

    imgCenaOriginal = cv2.imread("imagens/teste.jpg")

    if imgCenaOriginal is None:
        print ("\nErro: Arquivo de imagem não lido\n\n")
        os.system("pause")
        return

    listaDePossiveisPlacas = DetectarPlacas.DetectarPlacasInScene(imgCenaOriginal)

    listaDePossiveisPlacas = DetectarCaracteres.DetectarCaracteresNasPlacas(listaDePossiveisPlacas)

    cv2.imshow("imgCenaOriginal", imgCenaOriginal)

    if len(listaDePossiveisPlacas) == 0:
        print ("\nNenhuma placa foi encontrada\n")
    else:
        listaDePossiveisPlacas.sort(key=lambda possivelPlaca: len(possivelPlaca.strCaracteres), reverse=True)

        licPlaca = listaDePossiveisPlacas[0]

        cv2.imshow("imgPlaca", licPlaca.imgPlaca)
        cv2.imshow("imgThreshold", licPlaca.imgThreshold)

        if len(licPlaca.strCaracteres) == 0:
            print ("\nNenhum caractere foi encontrado\n\n")
            return

        desenharRetanguloVermelhoAoRedorDaPlaca(imgCenaOriginal, licPlaca)

        print ("\nPlaca lida da imagem = " + licPlaca.strCaracteres + "\n")
        print ("----------------------------------------")

        escreverCaracteresDaPlacaNaImagem(imgCenaOriginal, licPlaca)

        cv2.imshow("imgCenaOriginal", imgCenaOriginal)

        cv2.imwrite("imgCenaOriginal.png", imgCenaOriginal)

    cv2.waitKey(0)

    return


def desenharRetanguloVermelhoAoRedorDaPlaca(imgCenaOriginal, licPlaca):
    p2fRectPoints = np.array(cv2.boxPoints(licPlaca.rrLocationOfPlacaInScene))

    pointsPlaca = p2fRectPoints.astype('int')

    cv2.line(imgCenaOriginal, tuple(pointsPlaca[0]), tuple(pointsPlaca[1]), ESCALA_VERMELHO, 2)
    cv2.line(imgCenaOriginal, tuple(pointsPlaca[1]), tuple(pointsPlaca[2]), ESCALA_VERMELHO, 2)
    cv2.line(imgCenaOriginal, tuple(pointsPlaca[2]), tuple(pointsPlaca[3]), ESCALA_VERMELHO, 2)
    cv2.line(imgCenaOriginal, tuple(pointsPlaca[3]), tuple(pointsPlaca[0]), ESCALA_VERMELHO, 2)


def escreverCaracteresDaPlacaNaImagem(imgCenaOriginal, licPlaca):

    sceneHeight, sceneWidth, sceneNumChannels = imgCenaOriginal.shape
    PlacaHeight, PlacaWidth, PlacaNumChannels = licPlaca.imgPlaca.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(PlacaHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlaca.strCaracteres, intFontFace, fltFontScale, intFontThickness)

    ((intPlacaCenterX, intPlacaCenterY), (intPlacaWidth, intPlacaHeight),
     fltCorrectionAngleInDeg) = licPlaca.rrLocationOfPlacaInScene

    intPlacaCenterX = int(intPlacaCenterX)
    intPlacaCenterY = int(intPlacaCenterY)

    ptCenterOfTextAreaX = int(intPlacaCenterX)

    if intPlacaCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlacaCenterY)) + int(round(PlacaHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlacaCenterY)) - int(round(PlacaHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    cv2.putText(imgCenaOriginal, licPlaca.strCaracteres, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, ESCALA_AMARELO, intFontThickness)

if __name__ == "__main__":
    main()



















