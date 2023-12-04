# utils.py

from math import atan2, degrees, radians, sqrt, sin, cos

def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_heading(org, target):
    d_x = target.x - org.x
    d_y = target.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180

def remove_dead_organisms(organisms):
    return [org for org in organisms if org.alive]
