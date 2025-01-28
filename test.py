import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import torchaudio
import torchaudio.transforms
import random
import os
import tqdm


def random_room_dimensions():
    x = random.uniform(4, 10)
    y = random.uniform(4, 10)
    z = random.uniform(2.5, 5)
    return np.array([x, y, z])


def random_head_position(room_dim):
    rdx, rdy, _ = room_dim
    x = random.uniform(0.125*rdx, 0.875*rdx)
    y = random.uniform(0.125*rdy, 0.875*rdy)
    z = random.uniform(1, 1.75)
    return np.array([x, y, z])


def random_mouth_position(head_pos, head_ang):
    hpx, hpy, hpz = head_pos
    rdx = random.uniform(-0.01, 0.01)
    rdy = random.uniform(0.11, 0.15)
    rdz = random.uniform(0.02, 0.04)
    x = hpx + rdx*np.cos(head_ang) - rdy*np.sin(head_ang)
    y = hpy + rdx*np.sin(head_ang) + rdy*np.cos(head_ang)
    z = hpz - rdz
    return np.array([x, y, z])


def random_ears_position(head_pos, head_ang):
    hpx, hpy, hpz = head_pos
    rdx = random.uniform(0.08, 0.09)
    ear_center_l = np.array([hpx - rdx*np.cos(head_ang), hpy - rdx*np.sin(head_ang), hpz])
    ear_center_r = np.array([hpx + rdx*np.cos(head_ang), hpy + rdx*np.sin(head_ang), hpz])
    return np.array([ear_center_l, ear_center_r])


def define_mics_position(ears_pos):
    ear_center_l, ear_center_r = ears_pos
    lex, ley, lez = ear_center_l
    rex, rey, rez = ear_center_r
    mic_l_1 = np.array([lex, ley, lez - 0.01])
    mic_l_2 = np.array([lex, ley, lez + 0.01])    
    mic_r_1 = np.array([rex, rey, rez - 0.01])
    mic_r_2 = np.array([rex, rey, rez + 0.01])
    return np.array([mic_l_1, mic_l_2, mic_r_1, mic_r_2])


def random_distractor_position(room_dim, head_pos):
    rdx, rdy, _ = room_dim
    hpx, hpy, _ = head_pos
    angl = random.uniform(0, 2*np.pi)

    dx = np.cos(angl)*1
    if dx > 0:
        x_min = hpx + dx
        x_max = 0.875*rdx
        if x_min > x_max:
            x_min = 0.125*rdx
            x_max = hpx - dx     
    else:
        x_min = 0.125*rdx
        x_max = hpx + dx
        if x_min > x_max:
            x_min = hpx - dx
            x_max = 0.875*rdx
    x = random.uniform(x_min, x_max)

    dy = np.sin(angl)*1
    if dy > 0:
        y_min = hpy + dy
        y_max = 0.875*rdy
        if y_min > y_max:
            y_min = 0.125*rdy
            y_max = hpy - dy
    else:
        y_min = 0.125*rdy
        y_max = hpy + dy
        if y_min > y_max:
            y_min = hpy - dy
            y_max = 0.875*rdy
    y = random.uniform(y_min, y_max)
    
    z = random.uniform(1, 1.75)
    return np.array([x, y, z])

if __name__ == '__main__':

    room_dim = random_room_dimensions()
    head_pos = random_head_position(room_dim)
    head_ang = random.uniform(0, 2*np.pi)
    ears_pos = random_ears_position(head_pos, head_ang)
    mics_pos = define_mics_position(ears_pos)
    mouth_pos = random_mouth_position(head_pos,  head_ang)
    distractor_pos = random_distractor_position(room_dim, head_pos)    

    x = np.zeros(1000)

    for i in range(1000):
        room_dim = random_room_dimensions()
        head_pos = random_head_position(room_dim)
        distractor_pos = random_distractor_position(room_dim, head_pos)