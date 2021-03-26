import numpy as np 
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from visual_odometry import PinholeCamera, VisualOdometry
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
    
meta_data_path = './interpolated.csv'
base_path = "C:/Users/asus/Desktop/Self Driving/Comma Scratch/data"

skip_factor = 1
start = 8800

df = pd.read_csv(meta_data_path)
df = df[df['frame_id'] == 'center_camera'][::skip_factor][start:]
df['timestamp'] = df['timestamp']/1000000000
ins = pd.read_csv('imu.csv')
ins['timestamp'] = ins['timestamp']/1000000000

cam = PinholeCamera(640, 480, 2152.445406, 2166.161453, 268.010970, 302.594211)
vo = VisualOdometry(cam, meta_data_path)
model = Bicycle()

traj = np.zeros((600,600,3), dtype=np.uint8)
#rel_x, rel_y = df[['lat', 'long']].values[0]
prev_lat, prev_lon = df[['lat', 'long']].values[0]
xg, yg = 0, 0

R_y = 0.4
Q_y = 0.025
INIT_y = 0
INIT_VAR_y = 0.5**2
kalman_y = KalmanFilter(INIT_y, INIT_VAR_y, R_y**2, Q_y**2)

R_x = 0.6
Q_x = 0.02
INIT_x = 0
INIT_VAR_x = 0
kalman_x = KalmanFilter(INIT_x, INIT_VAR_x, R_x**2, Q_x**2)

bs = []
gs = []

N_STEPS = 300

counter = 0
trajectory = []
for img_id in range(len(df['filename'].values)):
        
    img_path = os.path.join(base_path, df['filename'].values[img_id])
    img = cv2.imread(img_path, 0)
    v, delta, current_time = df[['speed', 'angle', 'timestamp']].values[img_id]
    
    #acc_idx = abs(ins['timestamp'].values - current_time).argmin()
    #acc_x = ins['ax'].values[acc_idx]
    #acc_y = -ins['ay'].values[acc_idx]
    #print(v, delta)
    dy_b, dx_b = model.step(v, delta, current_time)
    kalman_y.predict(dy_b)
    kalman_x.predict(-dx_b)
    
    # Visual Odometery update
    vo.update(img, img_id)

    cur_t = vo.cur_t
    if(img_id > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
            x, y, z = 0., 0., 0.

    vo.x, vo.y, vo.z = x, y, z

    lat, lon = df[['lat', 'long']].values[img_id]
    d_xg, d_yg = latlon_to_xy_grid(lat, lon, prev_lat, prev_lon)
    
    xg += d_xg * 1000
    yg += d_yg * 1000

    if img_id < 2:
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
    

    trajectory.append([kalman_x.x, kalman_y.x, -model.yc, model.xc, yg_rot*sign, xg_rot, x, z, xg, yg])

    draw_xg, draw_yg = int(xg_rot + 90), int(yg_rot*sign + 290)
    draw_x, draw_y = int(x) + 290, int(z) + 90
    draw_xb, draw_yb = -int(model.yc) + 290, int(model.xc) + 90
    draw_kalman_x, draw_kalman_y = int(kalman_x.x) + 290, int(kalman_y.x) + 90
    #print(kalman_y.P, kalman_x.P)
    print(sign)

    #cv2.circle(traj, (draw_x, draw_y), 1, (255, 0, 0), 1)
    cv2.circle(traj, (draw_xb, draw_yb), 1, (0,0,255), 1)
    cv2.circle(traj, (draw_yg, draw_xg), 1, (0,255, 0), 1)
    cv2.circle(traj, (draw_kalman_x, draw_kalman_y), 1, (255, 255, 255), 1)
    
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)
    print(kalman_y.x)

    if counter > N_STEPS:
        break
    counter += 1

trajectory = np.array(trajectory)
cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(list(trajectory[:, 0][::2]), list(trajectory[:, 1][::2]), ds=0.1)
plt.scatter(trajectory[:, 0], trajectory[:, 1], label='Filtered', s=0.5)
plt.scatter(trajectory[:, 2], trajectory[:, 3], label='Bicycle Model', s=0.5)
plt.scatter(trajectory[:, 4], trajectory[:, 5], label='GNSS',  s=0.5)
plt.scatter(trajectory[:, 6], trajectory[:, 7], label='Visual Odometery',  s=0.5)
plt.scatter(cx, cy, label='Spline',  s=0.5)
plt.xticks([-3, -2, -1, 0, 1, 2, 3])
plt.legend(loc='best')
plt.show()




