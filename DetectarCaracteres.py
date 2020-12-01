import cv2
import numpy as np
import math
import random

import Main
import Preprocesso
import PossivelCaractere


kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_LARGURA = 2
MIN_PIXEL_ALTURA = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_MUDANCA_EM_AREA = 0.5

MAX_MUDANCA_NA_LARGURA = 0.8
MAX_MUDANCA_NA_ALTURA = 0.2

MAX_ANGULO_ENTRE_CARACTERES = 12.0

MIN_NUMERO_DE_COMBINACAO_CARACTERES = 3

REDIMENSIONAR_CHAR_IMAGEM_LARGURA = 20
REDIMENSIONAR_CHAR_IMAGEM_ALTURA = 30

MIN_CONTORNO_AREA = 100


def loadKNNDataAndTrainKNN():
    todosOsContornosComOsDados = []
    contornosValidosComData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return False

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest.setDefaultK(1)

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    return True

def DetectarCaracteresNasPlacas(listaDePossiveisPlacas):
    intPlacaCounter = 0
    imgContours = None
    contornos = []

    if len(listaDePossiveisPlacas) == 0:
        return listaDePossiveisPlacas

    for possivelPlaca in listaDePossiveisPlacas:

        possivelPlaca.imgEscalaDeCinza, possivelPlaca.imgThreshold = Preprocesso.Preprocesso(possivelPlaca.imgPlaca)

        if Main.mostrarPassos == True:
            cv2.imshow("5a", possivelPlaca.imgPlaca)
            cv2.imshow("5b", possivelPlaca.imgEscalaDeCinza)
            cv2.imshow("5c", possivelPlaca.imgThreshold)

        possivelPlaca.imgThreshold = cv2.resize(possivelPlaca.imgThreshold, (0, 0), fx=1.6, fy=1.6)

        thresholdValue, possivelPlaca.imgThreshold = cv2.threshold(possivelPlaca.imgThreshold, 0.0, 255.0,
                                                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.mostrarPassos == True:
            cv2.imshow("5d", possivelPlaca.imgThreshold)

        listaDePossiveisCaracteresInPlaca = encontrarPossivelCaractereNaPlaca(possivelPlaca.imgEscalaDeCinza,
                                                                              possivelPlaca.imgThreshold)

        if Main.mostrarPassos == True:
            altura, largura, numCanais = possivelPlaca.imgPlaca.shape
            imgContours = np.zeros((altura, largura, 3), np.uint8)
            del contornos[:]

            for possivelCaractere in listaDePossiveisCaracteresInPlaca:
                contornos.append(possivelCaractere.contour)

            cv2.drawContours(imgContours, contornos, -1, Main.ESCALA_BRANCO)

            cv2.imshow("6", imgContours)

        listaDeListasDeCombinacaoDeCaracteresInPlaca = findListOfListsOfMatchingCaracteres(
            listaDePossiveisCaracteresInPlaca)

        if Main.mostrarPassos == True:
            imgContours = np.zeros((altura, largura, 3), np.uint8)
            del contornos[:]

            for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInPlaca:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingCaractere in listaDeCombinacaoDeCaracteres:
                    contornos.append(matchingCaractere.contour)
                cv2.drawContours(imgContours, contornos, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            cv2.imshow("7", imgContours)

        if (len(
                listaDeListasDeCombinacaoDeCaracteresInPlaca) == 0):

            if Main.mostrarPassos == True:
                print ("chars encontrados na placa número " + str(
                    intPlacaCounter) + " = (none), clique em qualquer imagem e pressione uma tecla para continuar . . .")
                intPlacaCounter = intPlacaCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)

            possivelPlaca.strCaracteres = ""
            continue

        for i in range(0, len(
                listaDeListasDeCombinacaoDeCaracteresInPlaca)):
            listaDeListasDeCombinacaoDeCaracteresInPlaca[i].sort(key=lambda
                matchingCaractere: matchingCaractere.intCenterX)
            listaDeListasDeCombinacaoDeCaracteresInPlaca[i] = removerSobreposicaoDeCaracteres(
                listaDeListasDeCombinacaoDeCaracteresInPlaca[i])

        if Main.mostrarPassos == True:
            imgContours = np.zeros((altura, largura, 3), np.uint8)

            for listaDeCombinacaoDeCaracteres in listaDeListasDeCombinacaoDeCaracteresInPlaca:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contornos[:]

                for matchingCaractere in listaDeCombinacaoDeCaracteres:
                    contornos.append(matchingCaractere.contour)

                cv2.drawContours(imgContours, contornos, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            cv2.imshow("8", imgContours)

        intLenOfLongestListOfCaracteres = 0
        intIndexOfLongestListOfCaracteres = 0

        for i in range(0, len(listaDeListasDeCombinacaoDeCaracteresInPlaca)):
            if len(listaDeListasDeCombinacaoDeCaracteresInPlaca[i]) > intLenOfLongestListOfCaracteres:
                intLenOfLongestListOfCaracteres = len(listaDeListasDeCombinacaoDeCaracteresInPlaca[i])
                intIndexOfLongestListOfCaracteres = i

        maiorListaDeCaracteresCorrespondentesNaPlaca = listaDeListasDeCombinacaoDeCaracteresInPlaca[
            intIndexOfLongestListOfCaracteres]

        if Main.mostrarPassos == True:
            imgContours = np.zeros((altura, largura, 3), np.uint8)
            del contornos[:]

            for matchingCaractere in maiorListaDeCaracteresCorrespondentesNaPlaca:
                contornos.append(matchingCaractere.contour)

            cv2.drawContours(imgContours, contornos, -1, Main.ESCALA_BRANCO)

            cv2.imshow("9", imgContours)

        possivelPlaca.strCaracteres = recognizeCaracteresInPlaca(possivelPlaca.imgThreshold,
                                                                 maiorListaDeCaracteresCorrespondentesNaPlaca)

        if Main.mostrarPassos == True:
            print ("Caracteres encontrados no numero da placa " + str(
                intPlacaCounter) + " = " + possivelPlaca.strCaracteres + ", clique em qualquer imagem e pressione uma tecla para continuar . . .")
            intPlacaCounter = intPlacaCounter + 1
            cv2.waitKey(0)


    if Main.mostrarPassos == True:
        print (
        "\nDeteccaoo de caracteres completa, clique em qualquer imagem e pressione uma tecla para continuar . . .\n")
        cv2.waitKey(0)

    return listaDePossiveisPlacas

def encontrarPossivelCaractereNaPlaca(imgEscalaDeCinza, imgThreshold):
    listaDePossiveisCaracteres = []
    contornos = []
    imgThresholdCopia = imgThreshold.copy()

    contornos, npaHierarchy = cv2.findContours(imgThresholdCopia, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contornos:
        possivelCaractere = PossivelCaractere.PossivelCaractere(contour)

        if verificaSePossivelCaractere(
                possivelCaractere):
            listaDePossiveisCaracteres.append(possivelCaractere)

    return listaDePossiveisCaracteres


def verificaSePossivelCaractere(possivelCaractere):
    if (possivelCaractere.intBoundingRectArea > MIN_PIXEL_AREA and
                possivelCaractere.intBoundingRectWidth > MIN_PIXEL_LARGURA and possivelCaractere.intBoundingRectHeight > MIN_PIXEL_ALTURA and
                MIN_ASPECT_RATIO < possivelCaractere.fltAspectRatio and possivelCaractere.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

def findListOfListsOfMatchingCaracteres(listaDePossiveisCaracteres):
    listaDeListasDeCombinacaoDeCaracteres = []

    for possivelCaractere in listaDePossiveisCaracteres:
        listaDeCombinacaoDeCaracteres = encontrarListaDeCombincacaoDeCaracteres(possivelCaractere, listaDePossiveisCaracteres)

        listaDeCombinacaoDeCaracteres.append(possivelCaractere)

        if len(listaDeCombinacaoDeCaracteres) < MIN_NUMERO_DE_COMBINACAO_CARACTERES:
            continue

        listaDeListasDeCombinacaoDeCaracteres.append(listaDeCombinacaoDeCaracteres)

        listaDePossiveisCaracteresComAtualCombinacaoRemovida = []

        listaDePossiveisCaracteresComAtualCombinacaoRemovida = list(
            set(listaDePossiveisCaracteres) - set(listaDeCombinacaoDeCaracteres))

        recursiveListOfListsOfMatchingCaracteres = findListOfListsOfMatchingCaracteres(
            listaDePossiveisCaracteresComAtualCombinacaoRemovida)

        for listaRecursivaDeCombinacaoDeCaracteres in recursiveListOfListsOfMatchingCaracteres:
            listaDeListasDeCombinacaoDeCaracteres.append(listaRecursivaDeCombinacaoDeCaracteres)
        break

    return listaDeListasDeCombinacaoDeCaracteres

def encontrarListaDeCombincacaoDeCaracteres(possivelCaractere, listOfCaracteres):

    listaDeCombinacaoDeCaracteres = []

    for possivelCombinacaoDeCaractere in listOfCaracteres:
        if possivelCombinacaoDeCaractere == possivelCaractere:
            continue
        fltDistanciaEntreCaracteres = distanciaEntreCaracteres(possivelCaractere, possivelCombinacaoDeCaractere)

        fltAnguloEntreCaracteres = anguloEntreCaracteres(possivelCaractere, possivelCombinacaoDeCaractere)

        fltChangeInArea = float(
            abs(possivelCombinacaoDeCaractere.intBoundingRectArea - possivelCaractere.intBoundingRectArea)) / float(
            possivelCaractere.intBoundingRectArea)

        fltChangeInWidth = float(
            abs(possivelCombinacaoDeCaractere.intBoundingRectWidth - possivelCaractere.intBoundingRectWidth)) / float(
            possivelCaractere.intBoundingRectWidth)
        fltChangeInHeight = float(
            abs(possivelCombinacaoDeCaractere.intBoundingRectHeight - possivelCaractere.intBoundingRectHeight)) / float(
            possivelCaractere.intBoundingRectHeight)

        if (fltDistanciaEntreCaracteres < (possivelCaractere.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                    fltAnguloEntreCaracteres < MAX_ANGULO_ENTRE_CARACTERES and
                    fltChangeInArea < MAX_MUDANCA_EM_AREA and
                    fltChangeInWidth < MAX_MUDANCA_NA_LARGURA and
                    fltChangeInHeight < MAX_MUDANCA_NA_ALTURA):
            listaDeCombinacaoDeCaracteres.append(possivelCombinacaoDeCaractere)

    return listaDeCombinacaoDeCaracteres

def distanciaEntreCaracteres(primeiroCaractere, segundoCaractere):
    intX = abs(primeiroCaractere.intCenterX - segundoCaractere.intCenterX)
    intY = abs(primeiroCaractere.intCenterY - segundoCaractere.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

def anguloEntreCaracteres(primeiroCaractere, segundoCaractere):
    fltAdj = float(abs(primeiroCaractere.intCenterX - segundoCaractere.intCenterX))
    fltOpp = float(abs(primeiroCaractere.intCenterY - segundoCaractere.intCenterY))

    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)

    return fltAngleInDeg

def removerSobreposicaoDeCaracteres(listaDeCombinacaoDeCaracteres):
    listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved = list(listaDeCombinacaoDeCaracteres)

    for caractereAtual in listaDeCombinacaoDeCaracteres:
        for outroCaractere in listaDeCombinacaoDeCaracteres:
            if caractereAtual != outroCaractere:
                if distanciaEntreCaracteres(caractereAtual, outroCaractere) < (
                    caractereAtual.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if caractereAtual.intBoundingRectArea < outroCaractere.intBoundingRectArea:
                        if caractereAtual in listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved:
                            listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved.remove(caractereAtual)
                    else:
                        if outroCaractere in listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved:
                            listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved.remove(outroCaractere)

    return listaDeCombinacaoDeCaracteresWithInnerCaractereRemoved

def recognizeCaracteresInPlaca(imgThreshold, listaDeCombinacaoDeCaracteres):
    strCaracteres = ""

    altura, largura = imgThreshold.shape

    imgThresholdColor = np.zeros((altura, largura, 3), np.uint8)

    listaDeCombinacaoDeCaracteres.sort(key=lambda matchingCaractere: matchingCaractere.intCenterX)

    cv2.cvtColor(imgThreshold, cv2.COLOR_GRAY2BGR, imgThresholdColor)

    for caractereAtual in listaDeCombinacaoDeCaracteres:
        pt1 = (caractereAtual.intBoundingRectX, caractereAtual.intBoundingRectY)
        pt2 = ((caractereAtual.intBoundingRectX + caractereAtual.intBoundingRectWidth),
               (caractereAtual.intBoundingRectY + caractereAtual.intBoundingRectHeight))

        cv2.rectangle(imgThresholdColor, pt1, pt2, Main.ESCALA_VERDE, 2)

        imgROI = imgThreshold[
                 caractereAtual.intBoundingRectY: caractereAtual.intBoundingRectY + caractereAtual.intBoundingRectHeight,
                 caractereAtual.intBoundingRectX: caractereAtual.intBoundingRectX + caractereAtual.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (REDIMENSIONAR_CHAR_IMAGEM_LARGURA,
                                            REDIMENSIONAR_CHAR_IMAGEM_ALTURA))

        npaROIResized = imgROIResized.reshape((1,
                                               REDIMENSIONAR_CHAR_IMAGEM_LARGURA * REDIMENSIONAR_CHAR_IMAGEM_ALTURA))

        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)

        strCurrentCaractere = str(chr(int(npaResults[0][0])))

        strCaracteres = strCaracteres + strCurrentCaractere


    if Main.mostrarPassos == True:
        cv2.imshow("10", imgThresholdColor)

    return strCaracteres







