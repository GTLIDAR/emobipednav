import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import glob
import time
from math import cos, sin, hypot, fabs

n_laser = 1800

with_laser = True
complex_env = True

specific_file_name = './logs/sac_ae_digit_mujoco_digit_mujoco_203550_3/seed_1/final_test_episodes_mannul/eval_23_0.npz'

log_env = np.load(specific_file_name)

robot = log_env['robot']
steps = robot.shape[0]
humans = log_env['humans']
if with_laser:
    laser = log_env['laser']
    laser_beam = laser.shape[1]
human_num = humans.shape[1]
goal = log_env['goal']

static_obstacles = []
radius_robot = 0.3

draw_discomfort = False
static_discomfort = 0.2
emotion_discomfort = [0.2, 0.35, 0.5]

#####real time plot for simulation case########
radius = 0.3
plt.ion()
plt.show()
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(steps):
    if fabs(robot[i][0]) > 50.0:
        break
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)

    scan_intersection = []
    if with_laser:
        for laser_i in range(laser_beam):
            scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
    for human_i in range(human_num):
        if human_i == 1:
            human_color = 'm'
        else:
            human_color = 'b'
        human_circle = plt.Circle(humans[i][human_i, 0:2], humans[i][human_i, 2], fill=False, color=human_color)
        ax.add_artist(human_circle)
        
        
        human_circle_discomfort = plt.Circle(humans[i][human_i, 0:2], 
                                            radius + emotion_discomfort[int(humans[i][human_i, 3])], 
                                            fill=False, color=human_color, linestyle='--')
        ax.add_artist(human_circle_discomfort)
        
    ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius_robot, fill=True, color='r'))
    ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
    plt.text(-4.5, -4.5, str(round(i * 0.4, 2)), fontsize=20)
    print('action: ', robot[i][2], robot[i][3])
    if complex_env:
        static_obstacles = log_env['static_obstacles']
        static_obstacle_num = static_obstacles.shape[1]
        for static_i in range(static_obstacle_num):
            ax.add_artist(plt.Circle((static_obstacles[i][static_i, 0], static_obstacles[i][static_i, 1]), 
                                      static_obstacles[i][static_i, 2],
                                      fill=True, color='c'))
            ax.add_artist(plt.Circle(static_obstacles[i][static_i, 0:2], static_obstacles[i][static_i, 2] + static_discomfort, 
                                     fill=False, color='b', linestyle='--'))

    x, y, theta = robot[i][0], robot[i][1], robot[i][4]
    print('robot position: ', x, y)
    dx = cos(theta)
    dy = sin(theta)
    ax.arrow(x, y, dx, dy,
        width=0.01,
        length_includes_head=True, 
        head_width=0.15,
        head_length=1,
        fc='r',
        ec='r')
    
    if with_laser:
        ii = 0
        lines = []
        while ii < n_laser:
            lines.append(scan_intersection[ii])
            ii = ii + 36
        lc = mc.LineCollection(lines)
        ax.add_collection(lc)
    plt.draw()
    plt.pause(0.001)
    plt.cla()
    time.sleep(0.2)