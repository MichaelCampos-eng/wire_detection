"""
All Classes
===========

This set of classes contains all classes used in the package.

author: C. R. Kelly
email: CK598@cam.ac.uk

"""
from sympy import nsimplify
from sympy.matrices import Matrix
from sympy.polys.matrices import DomainMatrix
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
import math
import os
import cv2

from utils import binaryConversion, binarySkeleton
from config import Config

config = Config()


# Circuit Diagram Image Class
class Image:

    def __init__(self, image, path):
        self.name = os.path.splitext(os.path.split(path)[1])[0]
        self.image = image
        self.binaryImage = binaryConversion(self.image)
        self.binarySkeleton = binarySkeleton(self.image)
        self.cleanedImage = []
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.centre = (int(self.width / 2), int(self.height / 2))
        self.size = np.size(image)
        self.path = path

    # gets image border
    def getBorder(self):
        rows = np.any(self.image, axis=0)
        columns = np.any(self.image, axis=1)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(columns)[0][[0, -1]]
        imageBorder = [top, bottom, left, right]
        return imageBorder

    # Displays image
    def displayImage(self):
        plt.figure(figsize=(self.width, self.height))
        plt.clf()
        plt.imshow(self.image, cmap=cm.gray)
        return self

    # Displays threshold image
    def displayBinaryImage(self):
        plt.clf()
        plt.imshow(self.binaryImage, cmap=cm.gray)
        return self

    # Displays threshold skeleton image
    def displayBinarySkeleton(self):
        plt.clf()
        plt.imshow(binarySkeleton(self.image), cmap=cm.gray)
        return self

    # Plots found wires
    def plotWires(self, HorizWires, VertWires):
        img = self.image.copy()
        for wire in HorizWires:
            top, bottom, left, right = wire.line
            cv2.line(img, (left, top), (right, bottom), (255, 0, 0), 2)

        for wire in VertWires:
            top, bottom, left, right = wire.line
            cv2.line(img, (left, top), (right, bottom), (0, 255, 0), 2) 
        cv2.imwrite('output_image_with_wires.png', img)

    # Plots found junctions
    def plotJunctions(self, Junctions, Letters=True):

        for i in range(len(Junctions)):
            y, x = Junctions[i].centroid
            plt.scatter(x, y, c='g', s=50)
            if Letters:
                plt.text(x, y, s=Junctions[i].id, c='blue', size='xx-large')
        return self

    # Plots the blue bounding box of all detected ground symbols
    def plotGround(self, groundSymbols):

        for symbol in groundSymbols:
            top, bottom, left, right = symbol[0], symbol[1], symbol[2], symbol[3]

            plt.plot((left, right), (top, top), c='b')
            plt.plot((left, right), (bottom, bottom), c='b')
            plt.plot((left, left), (top, bottom), c='b')
            plt.plot((right, right), (top, bottom), c='b')

    # Plots the original image with marked crop region, and, the cropped image
    def plotLabels(self, labels, text=False):

        strings, boxes = labels
        plt.close('all')

        # Generating figure
        fig, axes = plt.subplots(ncols=2, figsize=(15, 6.5))
        ax = axes.ravel()

        ax[0].set_xlim((0, self.width))
        ax[0].set_ylim((self.height, 0))
        ax[0].set_title('Original Image')
        ax[1].set_title(self.path + ' | Cropped' + ' | Text = ' + str(text))

        # plotting box
        for idx, box in enumerate(boxes):
            char = strings[idx]
            top, bottom, left, right = box

            ax[0].plot((left, right), (top, top), c='r')
            ax[0].plot((left, right), (bottom, bottom), c='r')
            ax[0].plot((left, left), (top, bottom), c='r')
            ax[0].plot((right, right), (top, bottom), c='r')
            if text:
                ax[0].text(left + int((right - left) / 2), (top - 2), char, c='b')

        ax[0].imshow(self.image, cmap=cm.gray)
        ax[1].imshow(self.cleanedImage, cmap=cm.gray)

        plt.show()

# Hough Lines Class - Horizontal
class HorizontalLines:
    def __init__(self, x1, y1, x2, y2):
        self.line = x1, y1, x2, y2
        self.length = math.hypot(abs(x2 - x1), abs(y1 - y2))
        self.centre = float(self.length / 2)
        self.start = y1, x1
        self.end = y2, x2
        self.inBox = False
        self.inPair = False


# Hough Lines Class - Vertical
class VerticalLines:
    def __init__(self, x1, y1, x2, y2):
        self.line = x1, y1, x2, y2
        self.length = math.hypot(abs(x2 - x1), abs(y1 - y2))
        self.centre = float(self.length / 2)
        self.start = y1, x1
        self.end = y2, x2
        self.inBox = False
        self.inPair = False


# Horizontal Wires Class
class WireHoriz:
    def __init__(self, y1, y2, x1, x2, binaryImage):
        self.wire = binaryImage[y1:y2, x1:x2]
        self.length = binaryImage[y1:y2, x1:x2].shape[1]
        self.centre = int(x1 + ((x2 - x1) / 2))
        self.line = y1, y2, x1, x2
        self.start = y1, x1
        self.end = y2, x2
        self.junctionStart = False
        self.junctionEnd = False
        self.componentStart = False
        self.componentEnd = False

    def getBorder(self):
        rows = np.any(self.wire, axis=0)
        columns = np.any(self.wire, axis=1)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(columns)[0][[0, -1]]
        wireBorder = [top, bottom, left, right]
        return wireBorder


# Vertical Wires Class
class WireVert:
    def __init__(self, y1, y2, x1, x2, binaryImage):
        self.wire = binaryImage[y1:y2, x1:x2]
        self.length = binaryImage[y1:y2, x1:x2].shape[1]
        self.centre = int(y1 + ((y2 - y1) / 2))
        self.line = y1, y2, x1, x2
        self.start = y1, x1
        self.end = y2, x2
        self.junctionStart = False
        self.junctionEnd = False
        self.componentStart = False
        self.componentEnd = False

    def getBorder(self):
        rows = np.any(self.wire, axis=0)
        columns = np.any(self.wire, axis=1)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(columns)[0][[0, -1]]
        wireBorder = [top, bottom, left, right]
        return wireBorder


# Wire Junctions class
class WireJunctions:
    def __init__(self, centroid):
        self.id = ''
        self.id_node = ''
        self.centroid = centroid[0], centroid[1]
        self.directions = 'NIL'  # N, S, E, W ->> north is up, south is down, west/east are left/right
        self.type = 'NIL'  # Corner, tri junction, quad junction
        self.associatedHWires = []
        self.associatedVWires = []
        self.isNode = True
        self.connectedNodesH = []
        self.connectedNodesV = []