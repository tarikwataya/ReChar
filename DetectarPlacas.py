import cv2
import numpy as np
import math
import Main
import random

import Preprocesso
import DetectarCaracteres
import PossivelPlaca
import PossivelCaractere

PLACA_LARGURA_FATOR_PREENCHIMENTO = 1.3
PLACA_ALTURA_FATOR_PREENCHIMENTO = 1.5

def DetectarPlacasInScene(imgCenaOriginal):
    listaDePossiveisPlacas = []

    altura, largura, numCanais = imgCenaOriginal.shape

    imgEscalaDeCinzaScene = np.zeros((altura, largura, 1), np.uint8)
    imgThresholdScene = np.zeros((altura, largura, 1), np.uint8)
    imgContours = np.zeros((altura, largura, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.mostrarPassos == True:
        cv2.imshow("0", imgCenaOriginal)

    imgEscalaDeCinzaScene, imgThresholdScene = Preprocesso.Preprocesso(imgCenaOriginal)

    if Main.mostrarPassos == True:
        cv2.imshow("1a", imgEscalaDeCinzaScene)
        cv2.imshow("1b", imgThresholdScene)

    listaDePossiveisCaracteresInScene = findPossivelCaracteresInScene(imgThresholdScene)

    if Main.mostrarPassos == True:
        print ("step 2 - len(listaDePossiveisCaracteresInScene) = " + str(len(listaDePossiveisCaracteresInScene)))

        imgContours = np.zeros((altura, largura, 3), np.uint8)

        contornos = []

        for possivelCaractere in listaDePossiveisCaracteresInScene:
            contornos.append(possivelCaractere.contour)

        cv2.drawContours(imgContours, contornos, -1, Main.ESCALA_BRANCO)
        cv2.imshow("2b", imgContours)

    listaDeListasDeCombinacaoDeCaracteresInScene = DetectarCaracteres.findListOfListsOfMatchingCaracteres(listaDePossiveisCaracteresInScene)

    if Main.mostrarPassos == True:
        print ("step 3 - listaDeListasDeCombinacaoDeCaracteresInScene.Count = " + str(len(listaDeListasDeCombinacaoDeCaracteresInScene)))

        imgContours = np.zeros((altura, largura, 3), np.uint8)

        for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contornos = []

            for matchingCaractere in listaDeCombinacaoDeCaracteres:
                contornos.append(matchingCaractere.contour)

            cv2.drawContours(imgContours, contornos, -1, (intRandomBlue, intRandomGreen, intRandomRed))

        cv2.imshow("3", imgContours)

    for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInScene:
        possivelPlaca = extrairPlaca(imgCenaOriginal, listaDeCombinacaoDeCaracteres)

        if possivelPlaca.imgPlaca is not None:
            listaDePossiveisPlacas.append(possivelPlaca)

    print ("\n" + str(len(listaDePossiveisPlacas)) + " possíveis placas encontrados")

    if Main.mostrarPassos == True:
        print ("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listaDePossiveisPlacas)):
            p2fRectPoints = cv2.boxPoints(listaDePossiveisPlacas[i].rrLocationOfPlacaInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.ESCALA_VERMELHO, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.ESCALA_VERMELHO, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.ESCALA_VERMELHO, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.ESCALA_VERMELHO, 2)

            cv2.imshow("4a", imgContours)

            print ("possível Placa " + str(i) + ", clique em qualquer imagem e pressione uma tecla para continuar. . .")

            cv2.imshow("4b", listaDePossiveisPlacas[i].imgPlaca)
            cv2.waitKey(0)

        print ("\ndetecção de Placa completa, clique em qualquer imagem e pressione uma tecla para iniciar o reconhecimento de caractere . . .\n")
        cv2.waitKey(0)

    return listaDePossiveisPlacas

def findPossivelCaracteresInScene(imgThreshold):
    listaDePossiveisCaracteres = []

    intCountOfPossivelCaracteres = 0

    imgThresholdCopia = imgThreshold.copy()

    contornos, npaHierarchy = cv2.findContours(imgThresholdCopia, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    altura, largura = imgThreshold.shape
    imgContours = np.zeros((altura, largura, 3), np.uint8)

    for i in range(0, len(contornos)):

        if Main.mostrarPassos == True:
            cv2.drawContours(imgContours, contornos, i, Main.ESCALA_BRANCO)

        possivelCaractere = PossivelCaractere.PossivelCaractere(contornos[i])

        if DetectarCaracteres.verificaSePossivelCaractere(possivelCaractere):
            intCountOfPossivelCaracteres = intCountOfPossivelCaracteres + 1
            listaDePossiveisCaracteres.append(possivelCaractere)

    if Main.mostrarPassos == True:
        print ("\netapa 2 - len(contornos) = " + str(len(contornos)))
        print ("etapa 2 - intCountOfPossivelCaracteres = " + str(intCountOfPossivelCaracteres))
        cv2.imshow("2a", imgContours)

    return listaDePossiveisCaracteres


def extrairPlaca(imgOriginal, listaDeCombinacaoDeCaracteres):
    possivelPlaca = PossivelPlaca.PossivelPlaca()

    listaDeCombinacaoDeCaracteres.sort(key = lambda matchingCaractere: matchingCaractere.intCenterX)

    fltPlacaCenterX = (listaDeCombinacaoDeCaracteres[0].intCenterX + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterX) / 2.0
    fltPlacaCenterY = (listaDeCombinacaoDeCaracteres[0].intCenterY + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterY) / 2.0

    ptPlacaCenter = fltPlacaCenterX, fltPlacaCenterY

    intPlacaWidth = int((listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intBoundingRectX + listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intBoundingRectWidth - listaDeCombinacaoDeCaracteres[0].intBoundingRectX) * PLACA_LARGURA_FATOR_PREENCHIMENTO)

    intTotalOfCaractereHeights = 0

    for matchingCaractere in listaDeCombinacaoDeCaracteres:
        intTotalOfCaractereHeights = intTotalOfCaractereHeights + matchingCaractere.intBoundingRectHeight

    fltAverageCaractereHeight = intTotalOfCaractereHeights / len(listaDeCombinacaoDeCaracteres)

    intPlacaHeight = int(fltAverageCaractereHeight * PLACA_ALTURA_FATOR_PREENCHIMENTO)

    fltOpposite = listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1].intCenterY - listaDeCombinacaoDeCaracteres[0].intCenterY
    fltHypotenuse = DetectarCaracteres.distanciaEntreCaracteres(listaDeCombinacaoDeCaracteres[0], listaDeCombinacaoDeCaracteres[len(listaDeCombinacaoDeCaracteres) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    possivelPlaca.rrLocationOfPlacaInScene = ( tuple(ptPlacaCenter), (intPlacaWidth, intPlacaHeight), fltCorrectionAngleInDeg )

    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlacaCenter), fltCorrectionAngleInDeg, 1.0)

    altura, largura, numCanais = imgOriginal.shape

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (largura, altura))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlacaWidth, intPlacaHeight), tuple(ptPlacaCenter))

    possivelPlaca.imgPlaca = imgCropped

    return possivelPlaca




