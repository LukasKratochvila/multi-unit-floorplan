import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from PIL import Image, ImageDraw

def draw(points):
    coord = []
    for point in points.split(' ')[:-1]:
        if point == '':
            continue
        x, y = point.split(',')
        x = float(x)
        y = float(y)
        coord.append([x, y])

    coord.append(coord[0])
    return coord


def circle(circle):
    return [(float(circle['cx']) - float(circle['r']), float(circle['cy']) - float(circle['r'])),
            (float(circle['cx']) + float(circle['r']), float(circle['cy']) + float(circle['r']))]


def parseFloor(floor, elements):
    for child in floor[0]:
        if 'id' in child.attrib:
            if child.attrib['id'] == 'Railing':
                assert len(child) == 1  # Railing consists of a single polygon
                assert 'points' in child[0].attrib
                elements[child.attrib['id']].append(draw(child[0].attrib['points']))
            elif child.attrib['id'] == 'Wall':
                elements[child.attrib['id']].append(draw(child[0].attrib['points']))
                for c in child[1:]:
                    if c.attrib['class'] == 'WallPropertyControl':
                        continue
                    assert c.attrib['id'] == 'Door' or c.attrib['id'] == 'Window'
                    elements[c.attrib['id']].append(draw(c[0].attrib['points']))
            elif child.attrib['id'] == 'Column':
                assert len(child) == 1  # Column consists of a single polygon or circle
                if 'Circle' in child.attrib['class']:
                    # Handle circles
                    assert 'cx' in child[0].attrib
                    elements[child.attrib['id']].append(circle(child[0].attrib))
                else:
                    assert 'points' in child[0].attrib
                    elements['Wall'].append(draw(child[0].attrib['points']))
            elif child.attrib['id'] == 'Stairs':
                for c in child:
                    elements[child.attrib['id']].append(draw(c[0].attrib['points']))
            else:
                # print(child.attrib['id'])
                pass


def parse_cubi_label(path):
    xml = ET.parse(os.path.join(path, 'model.svg'))
    root = xml.getroot()

    classes = ['Stairs', 'Railing', 'Wall', 'Window', 'Door', 'Column']
    elements = {}
    for c in classes:
        elements[c] = []

    for floor in root.find('{http://www.w3.org/2000/svg}g'):
        if floor.attrib['class'] == 'Floor':
            parseFloor(floor, elements)

    h, w, _ = cv2.imread(path + '/F1_scaled.png').shape

    for c in classes:
        bg = Image.new('L', size=(w, h))
        image_draw = ImageDraw.Draw(bg)
        for e in elements[c]:
            if len(e) == 2:  # Circle
                image_draw.ellipse(e, fill='white', outline='white', width=0)
            else:  # Polygon
                image_draw.polygon([(xy[0], xy[1]) for xy in e], fill='white', outline='white', width=0)
        bg.save(os.path.join(path, c.lower() + '.png'))

if __name__ == '__main__':
    parse_cubi_label('/media/kratochvila/Data/Datasets/cubicasa5k/colorful/30')
