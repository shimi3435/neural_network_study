from apng import APNG
from math import cos, pi, sin
from PIL import Image, ImageDraw

def make_animation(index, qval, pval):
    filename = "{:0>4}.png".format(index)
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    x = 30*sin(qval) + 50
    y = 30*cos(qval) + 50
    draw.line((50, 50, x, y), fill=(0, 255, 0), width=2)
    draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 0, 255))
    im.save(filename)
    return filename