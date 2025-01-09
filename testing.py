import numpy as np
from pdf2image import convert_from_path
import cv2
from classes import HorizontalLines, VerticalLines, WireHoriz, WireVert, WireJunctions
from skimage.feature import peak_local_max, match_template
from skimage.transform import probabilistic_hough_line, rotate
from io_1 import importImage
from tqdm import tqdm

def junctionDetection(HorizWires, VertWires):
    """ Detects which wires intersect to form a wire junction.

    :param HorizWires: list: List of horizontal wires returned by the HorizWires class.
    :param VertWires: list: List of vertical wires returned by the Vert class.
    :return list: Returns a list of found junctions
    """

    maxDistance = 5
    maxDistance = maxDistance ** 2
    foundJunctions = []
    id = 65  # ASCII VALUE
    for horizWire in HorizWires:

        for vertWire in VertWires:

            ###### Corner junctions
            hy, hx = horizWire.start
            vy, vx = vertWire.start
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.start[0], horizWire.start[1] - 1
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionStart = True
                vertWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hy, hx = horizWire.start
            vy, vx = vertWire.end
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.start[0], horizWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionStart = True
                vertWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hy, hx = horizWire.end
            vy, vx = vertWire.start
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.end[0], horizWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionEnd = True
                vertWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hy, hx = horizWire.end
            vy, vx = vertWire.end
            distance = (hx - vx) ** 2 + (hy - vy) ** 2
            if distance <= maxDistance:
                centroid = horizWire.end[0], horizWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 2
                Junction.type = 'Corner'
                horizWire.junctionEnd = True
                vertWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            ###### Tri Junctions east & west
            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hstart[1] - vstart[1])
            if vstart[0] < hstart[0] < vend[0] and distance <= maxDistance:
                centroid = horizWire.start[0], horizWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                horizWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hend[1] - vend[1])
            if vstart[0] < hend[0] < vend[0] and distance <= maxDistance:
                centroid = horizWire.end[0], horizWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                horizWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            ###### Tri Junctions north & south
            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hstart[0] - vstart[0])
            if hstart[1] < vstart[1] < hend[1] and distance <= maxDistance:
                centroid = vertWire.start[0], vertWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                vertWire.junctionStart = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            distance = abs(hend[0] - vend[0])
            if hstart[1] < vend[1] < hend[1] and distance <= maxDistance:
                centroid = vertWire.end[0], vertWire.end[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Tri'
                vertWire.junctionEnd = True
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)

            ###### Quad
            hstart, hend = horizWire.start, horizWire.end
            vstart, vend = vertWire.start, vertWire.end
            if vstart[0] < hstart[0] < vend[0] and hstart[1] < vstart[1] < hend[1]:
                centroid = horizWire.start[0], vertWire.start[1]
                Junction = WireJunctions(centroid)
                Junction.terminals = 3
                Junction.type = 'Quad'
                Junction.associatedHWires.append(horizWire)
                Junction.associatedVWires.append(vertWire)
                foundJunctions.append(Junction)
    for junction in foundJunctions:
        junction.id = chr(id)
        id += 1
    return foundJunctions



def sortWiresHough(horizWires, vertWires, image):
    """ Bubble sort for found wires. Sorts horiz wires so they are sorted from left to right first, then bottom to top. Vert wires in reverse

    :param horizWires: list: List of horizontal wires
    :param vertWires: list: List of vertical wires
    :param image: class: Image class of circuit diagram
    :return list: List of sorted horizontal wires
    :return list: List of sorted vertical wires
    """
    horizWiresSorted = []
    vertWiresSorted = []
    for height in range(image.height):
        horizLineSort = []
        for horizWire in horizWires:
            if horizWire.start[0] == height:
                horizLineSort.append(horizWire)
        for _ in horizLineSort:
            for horizLocation in range(len(horizLineSort) - 1):
                if horizLineSort[horizLocation].start[1] > horizLineSort[horizLocation + 1].start[1]:
                    horizLineSort[horizLocation], horizLineSort[horizLocation + 1] = horizLineSort[horizLocation + 1], \
                                                                                     horizLineSort[horizLocation]
        for horizSortedLine in horizLineSort:
            horizWiresSorted.append(horizSortedLine)

    for width in range(image.width):
        vertLineSort = []
        for vertWire in vertWires:
            if vertWire.start[1] == width:
                vertLineSort.append(vertWire)
        for __ in vertLineSort:
            for vertLocation in range(len(vertLineSort) - 1):
                if vertLineSort[vertLocation].start[0] > vertLineSort[vertLocation + 1].start[0]:
                    vertLineSort[vertLocation], vertLineSort[vertLocation + 1] = vertLineSort[vertLocation + 1], \
                                                                                 vertLineSort[vertLocation]
        for vertSortedLine in vertLineSort:
            vertWiresSorted.append(vertSortedLine)

    return horizWiresSorted, vertWiresSorted


def sortLines(lines):
    """ Sorts lines into horizontal and vertical lists.

    :param lines: list: List of lines returned by houghs probablistic transform.

    :return: list, list: Separate lists of horizontal and vertical lines.
    """
    horizLines = []
    vertLines = []

    for i in range(len(lines)):
        start, end = lines[i]
        x1, y1 = start
        x2, y2 = end
        if (y1 - y2) == 0:
            line = HorizontalLines(x1, y1, x2, y2)
            horizLines.append(line)
        elif (x1 - x2) == 0:
            line = VerticalLines(x1, y1, x2, y2)
            vertLines.append(line)

    return horizLines, vertLines


def wireDetect(border1, border2, wire, threshold=0.95):
    """ Detecting whether a wire is present based on a set of parameters

    :param border1: ndarray: Cropped segment of left/top side of the wire.
    :param border2: ndarray: Cropped segment of right/bottom side of the wire.
    :param wire:    ndarray: Cropped segment of the pixels representing the wire.
    :param threshold:   float: % of border pixels that can be filled for border to be counted as empty space.
    :return: bool:  True if wire is found to be present, else False.
    """
    border1Size = np.size(border1)
    border2Size = np.size(border2)
    wireSize = np.size(wire)
    b1Sum = float(np.sum(border1))
    b2Sum = float(np.sum(border2))
    wireSum = float(np.sum(wire))

    """
    Michael: Added the wireSum in conditional statement
    """

    # if b1Sum / border1Size <= threshold and b2Sum / border2Size <= threshold and wireSum / wireSize == 1:
    if border1Size > 0 and border2Size > 0 and wireSize > 30:
        if b1Sum / border1Size <= threshold and b2Sum / border2Size <= threshold:
            return True
        else:
            return False
    else:
        return False


def wireCheck(Wires, Wire):
    delta = 2
    duplicateWire = False
    wires = []
    for wire in Wires:
        wires.append(wire.line)
    for wire in wires:
        if abs(wire[0] - Wire.line[0]) <= delta and abs(wire[1] - Wire.line[1]) <= delta and abs(
                wire[2] - Wire.line[2]) <= delta and abs(wire[3] - Wire.line[3]) <= delta:
            duplicateWire = True
    return duplicateWire


def wireScanHough(image, minWireLength=10, borderSize=15):
    """ Scans for wires using Hough's transform

    :param binarySkeleton: ndarray: Binarised skeletonized image of circuit diagram
    :param minWireLength: int: minimum length before found line is counted as a wire
    :param borderSize: int: amount of empty space in pixels at the normal of the found line in both directions before  line is counted as wire
    :return: ndarray: Binary converted image.
    """
    HorizWires = []
    VertWires = []
    for loop in tqdm(range(0, 100), desc= "Hough transmation iter: "):
        angles = np.linspace(0, np.pi / 2, 2)
        lines = probabilistic_hough_line(image.binarySkeleton, threshold=10, line_length=minWireLength,
                                         line_gap=1,
                                         theta=angles)  # finding lines in the image using houghs transform # thresh was 35 for siren
        horizLines, vertLines = sortLines(lines)  # sorting found lines into horizontal and vertical categories

        for line in horizLines:
            left = line.start[1]
            right = line.end[1]

            if line.start[0] - borderSize <= 0:
                top = 0
                bottom = line.start[0] + borderSize
            elif line.start[0] + borderSize >= image.height:
                top = line.start[0] - borderSize
                bottom = image.height
            else:
                top, bottom, = line.start[0] - borderSize, line.start[0] + borderSize

            wire = image.binarySkeleton[line.start[0]:line.start[0] + 1, left:right]
            border1 = image.binarySkeleton[top:line.start[0], left:right]
            border2 = image.binarySkeleton[line.start[0] + 1:bottom, left:right]

            wirePresent = wireDetect(border1, border2, wire)
            if wirePresent:
                wire = WireHoriz(line.start[0], line.start[0], left, right, image.binarySkeleton)
                if not wireCheck(HorizWires, wire):
                    HorizWires.append(wire)

        for line in vertLines:
            bottom = line.start[0]
            top = line.end[0]

            if line.start[1] - borderSize <= 0:
                left = 0
                right = line.start[1] + borderSize
            elif line.start[1] + borderSize >= image.width:
                left = line.start[1] - borderSize
                right = image.width
            else:
                left, right, = line.start[1] - borderSize, line.start[1] + borderSize

            wire = image.binarySkeleton[top:bottom, line.start[1]:line.start[1] + 1]
            border1 = image.binarySkeleton[top:bottom, left:line.start[1]]
            border2 = image.binarySkeleton[top:bottom, line.start[1] + 1:right]

            wirePresent = wireDetect(border1, border2, wire)
            if wirePresent:
                wire = WireVert(top, bottom, line.start[1], line.start[1], image.binarySkeleton)
                if not wireCheck(VertWires, wire):
                    VertWires.append(wire)
    HorizWires, VertWires = sortWiresHough(HorizWires, VertWires, image)
    return HorizWires, VertWires


def wireAdd(start, end, HorizWires, VertWires, image):
    """ Manually add a wire, debug function

    :param start: tuple: (x, y) coordinates start of the wire
    :param end: tuple: (x, y) coordinates end of the wire
    :param HorizWires: list: List of horizontal wires returned by the HorizWires class.
    :param VertWires: list: List of vertical wires returned by the Vert class.
    :param image: ndarray: Image of circuit schematic
    """
    y1 = start[0]
    x1 = start[1]
    y2 = end[0]
    x2 = end[1]

    if y1 == y2:
        wire = WireHoriz(y1, y2, x1, x2, image.binarySkeleton)
        HorizWires.append(wire)
    elif x1 == x2:
        wire = WireVert(y1, y2, x1, x2, image.binarySkeleton)
        VertWires.append(wire)
    else:
        print("wire is neither horizontal nor vertical, check input")
        return

def plotWires(img, HorizWires, VertWires):
    for wire in HorizWires:
        top, bottom, left, right = wire.line
        cv2.line(img, (left, top), (right, bottom), (128, 0, 128), 2)

    for wire in VertWires:
        top, bottom, left, right = wire.line
        cv2.line(img, (left, top), (right, bottom), (0, 255, 0), 2) 
    return img

def plotJunctions(img, junctions):
    for i in range(len(junctions)):
        cv2.circle(img, junctions[i].centroid, 20, (0, 255, 0), -1)
    return img

def plotAll(img, junctions, HW, VW):
    img = plotWires(img, HW, VW)
    img = plotJunctions(img, junctions)
    cv2.imwrite('output_image_with_wires.png', img)

def removeLoops(HorizWires, VertWires):
    pass


if __name__ == "__main__":
    image = importImage("output_page_2.png")
    HorizWires, VertWires = wireScanHough(image)
    # junctions = junctionDetection(HorizWires, VertWires)
    cv_image = cv2.imread("output_page_2.png")
    img = plotWires(cv_image, HorizWires, VertWires)
    cv2.imwrite('ok.png', img)