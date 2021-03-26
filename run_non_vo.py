import numpy as np 
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from bicycle_model import Bicycle
from kalman_filter import KalmanFilter
import math
import cubic_spline_planner

def latlon_to_xy_grid(lat1, lon1, lat2, lon2): 
    dx = (lon1-lon2)*40075*math.cos((lat1+lat2)*math.pi/360)/360
    dy = (lat1-lat2)*40075/360
    return dx, dy

def rotate(x, y, rad):
    x_rot = math.cos(rad)*x - math.sin(rad)*y
    y_rot = math.sin(rad)*x + math.cos(rad)*y
    return x_rot, y_rot

def return_polyfit_line(points):

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    x_needed = []
    y_needed = np.arange(0, 100, 1) 
    for i in y_needed:
        idx = abs(y - i).argmin()
        x_needed.append(x[idx])
    return x_needed, y_needed
    
meta_data_path = './interpolated.csv'
base_path = "C:/Users/asus/Desktop/Self Driving/Comma Scratch/data"

MAX_METERS = 100
MAX_FRAMES = 300

skip_factor = 1
start = 200

df = pd.read_csv(meta_data_path)
df = df[df['frame_id'] == 'center_camera'][::skip_factor][start:]
df['timestamp'] = df['timestamp']/1000000000
ins = pd.read_csv('imu.csv')
ins['timestamp'] = ins['timestamp']/1000000000

prev_lat, prev_lon = df[['lat', 'long']].values[0]
xg, yg = 0, 0

R_y = 0.4
Q_y = 0.025
INIT_y = 0
INIT_VAR_y = 0.5**2

R_x = 0.55
Q_x = 0.02
INIT_x = 0
INIT_VAR_x = 0

dataset = []
img_tags = []

for img_id in range(int(8800 - start)):
    if 4930 < img_id + start < 5200:
        continue
    points = []
    counter = 0
    xg, yg = 0, 0
    prev_lat, prev_lon = df[['lat', 'long']].values[img_id]
    checkpoint = 0
    kalman_y = KalmanFilter(INIT_y, INIT_VAR_y, R_y**2, Q_y**2)
    kalman_x = KalmanFilter(INIT_x, INIT_VAR_x, R_x**2, Q_x**2)
    model = Bicycle()
    flag = True
    
    while checkpoint != 100:
        v, delta, current_time = df[['speed', 'angle', 'timestamp']].values[img_id + counter]
        dy_b, dx_b = model.step(v, delta, current_time)
        kalman_y.predict(dy_b)
        kalman_x.predict(-dx_b)

        lat, lon = df[['lat', 'long']].values[img_id + counter]
        d_xg, d_yg = latlon_to_xy_grid(lat, lon, prev_lat, prev_lon)

        xg += d_xg * 1000
        yg += d_yg * 1000

        if counter < 2:
            counter += 1
            angle = np.arctan2(xg, yg)
            if np.rad2deg(angle) < 0:
                sign = -1
            else:
                sign = 1
            continue

        prev_lat = lat
        prev_lon = lon

        xg_rot, yg_rot = rotate(xg, yg, -(np.pi/2 - angle))

        kalman_y.update(xg_rot)
        kalman_x.update(yg_rot*sign)

        checkpoint = int(kalman_y.x) 
        points.append([kalman_x.x, kalman_y.x])

        if counter > MAX_FRAMES:
            flag = False
            break
        
        counter += 1

    if flag:
        x, y = return_polyfit_line(points)
        dataset.append(x)
        img_tags.append(img_id + start)

        if img_id % 50  == 0:
            print('50 sequences added')

        if img_id % 300  == 0:
            np.save('trajectories', np.array(dataset))
            np.save('img_tags', np.array(img_tags))
            print('Checkpoint reached')
            print(img_id / (8800 - start), '% completed')




#14170
