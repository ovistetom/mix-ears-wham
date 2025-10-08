import numpy as np
import pyroomacoustics as pra
import anf_generator as anf
import random
import itertools 
import torch
import torchaudio
import torchaudio.transforms as tt
import os


SR = 16000
NFFT = 1024
NUM_CHANNELS = 4


def random_room_dimensions():
    x = random.randint(4, 10)
    y = random.randint(4, 10)
    z = random.randint(3, 5)
    return np.array([x, y, z])

def random_head_position(room_dim, mdtw=1):
    rdx, rdy, _ = room_dim
    # `mdtw`: minimum distance to the wall.
    x = random.uniform(mdtw, rdx - mdtw)
    y = random.uniform(mdtw, rdy - mdtw)
    z = random.uniform(1, 2)
    return np.array([x, y, z])

def random_head_yaw():
    return random.uniform(0, 2*np.pi)

def random_head_pitch():
    return random.uniform(-0.25*np.pi, 0.25*np.pi)

def random_head_roll():
    return random.uniform(-0.125*np.pi, 0.125*np.pi)

def random_mouth_position(head_pos, head_yaw=0.0, head_pitch=0.0, head_roll=0.0):
    # Define initial mouth position relative to head center.
    rdx = random.uniform(0.11, 0.15)
    rdy = random.uniform(-0.01, 0.01)
    rdz = random.uniform(-0.04, -0.02)
    mouth_pos = np.array([rdx, rdy, rdz])
    # Define rotation matrix around vertical axis (yaw).
    rot_yaw = np.array([[np.cos(head_yaw), -np.sin(head_yaw), 0.0], [np.sin(head_yaw), np.cos(head_yaw), 0.0], [0.0, 0.0, 1.0]])
    # Define rotation matrix around lateral axis (pitch).
    rot_pitch = np.array([[np.cos(head_pitch), 0.0, np.sin(head_pitch)], [0.0, 1.0, 0.0], [-np.sin(head_pitch), 0.0, np.cos(head_pitch)]])
    # Define rotation matrix around horizontal axis (roll).
    rot_roll = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(head_roll), -np.sin(head_roll)], [0.0, np.sin(head_roll), np.cos(head_roll)]])
    # Rotate mouth position.
    mouth_pos = np.einsum('ik, i -> k', rot_roll @ rot_pitch @ rot_yaw, mouth_pos)
    # Translate position of mouth to head center.
    mouth_pos = mouth_pos + head_pos
    return mouth_pos

def random_mics_position(head_pos, head_yaw=0.0, head_pitch=0.0, head_roll=0.0):
    rdy = random.uniform(0.08, 0.09)
    # Define initial position of mics relative to head center.
    mic_l_dn = np.array([0.0, + rdy, - 0.01])
    mic_l_up = np.array([0.0, + rdy, + 0.01])    
    mic_r_dn = np.array([0.0, - rdy, - 0.01])
    mic_r_up = np.array([0.0, - rdy, + 0.01])
    mics_pos = np.array([mic_l_up, mic_r_up, mic_l_dn, mic_r_dn])
    # Define rotation matrix around vertical axis (yaw).
    rot_yaw = np.array([[np.cos(head_yaw), -np.sin(head_yaw), 0.0], [np.sin(head_yaw), np.cos(head_yaw), 0.0], [0.0, 0.0, 1.0]])
    # Define rotation matrix around lateral axis (pitch).
    rot_pitch = np.array([[np.cos(head_pitch), 0.0, np.sin(head_pitch)], [0.0, 1.0, 0.0], [-np.sin(head_pitch), 0.0, np.cos(head_pitch)]])
    # Define rotation matrix around horizontal axis (roll).
    rot_roll = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(head_roll), -np.sin(head_roll)], [0.0, np.sin(head_roll), np.cos(head_roll)]])
    # Rotate position of mics.
    mics_pos = np.einsum('ik, mi -> mk', rot_roll @ rot_pitch @ rot_yaw, mics_pos)
    # Translate position of mics to head center.
    mics_pos = mics_pos + np.repeat(head_pos[None, ...], repeats=NUM_CHANNELS, axis=0)
    return mics_pos

def random_distractor_position(room_dim, head_pos, mdtw=1):
    rdx, rdy, _ = room_dim
    hpx, hpy, _ = head_pos
    angle = random.uniform(0, 2*np.pi)

    dx = np.cos(angle)*1
    if dx > 0:
        x_min = hpx + dx
        x_max = rdx - mdtw # (1-rdtw)*rdx
        if x_min > x_max:
            x_min = mdtw # rdtw*rdx
            x_max = hpx - dx     
    else:
        x_min = mdtw # rdtw*rdx
        x_max = hpx + dx
        if x_min > x_max:
            x_min = hpx - dx
            x_max = rdx - mdtw # (1-rdtw)*rdx
    x = random.uniform(x_min, x_max)

    dy = np.sin(angle)*1
    if dy > 0:
        y_min = hpy + dy
        y_max = rdy - mdtw # (1-rdtw)*rdy
        if y_min > y_max:
            y_min = mdtw # rdtw*rdy
            y_max = hpy - dy
    else:
        y_min = mdtw # rdtw*rdy
        y_max = hpy + dy
        if y_min > y_max:
            y_min = hpy - dy
            y_max = rdy - mdtw # (1-rdtw)*rdy
    y = random.uniform(y_min, y_max)
    
    z = random.uniform(1, 2)
    return np.array([x, y, z])

def random_noise_source_position(room_dim, head_pos, mdtw=0.25, num_sources=16):
    rdx, rdy, rdz = room_dim
    hpx, hpy, hpz = head_pos
    list_positions = []
    while len(list_positions) < num_sources:
        x = random.uniform(mdtw, rdx-mdtw)
        y = random.uniform(mdtw, rdy-mdtw)
        z = random.uniform(mdtw, rdz-mdtw)
        distance = np.sqrt((x-hpx)**2 + (y-hpy)**2 + (z-hpz)**2)
        if distance > 1.0:
            list_positions.append(np.array([x, y, z]))
    return list_positions

def random_snr(a=-5, b=5):
    return random.uniform(a, b)

def random_rt60(room_dim, max_rt60=2.0):
    v = np.prod(room_dim)
    s = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
    c = pra.parameters._constants_default['c']
    sc = 24
    a = v*sc*np.log(10)/(c*s)
    return random.uniform(a, max_rt60)

def pad_signal(signal, pad_right=0):
    return np.pad(signal, pad_width=((0, 0), (0, pad_right)), mode='constant', constant_values=0.0)

def pad_and_adjust(signal, target_length):
    padding_length = target_length - signal.shape[0]
    pad_left = np.random.randint(low=0, high=padding_length)
    pad_right = padding_length - pad_left
    return np.pad(signal, pad_width=(pad_left, pad_right), mode='constant', constant_values=0.0)

def force_noise_coherence(signal_noise, mics_pos, sr=SR, nfft=NFFT):
    # Define target spatial coherence.
    params = anf.CoherenceMatrix.Parameters(mic_positions=mics_pos, sc_type='spherical', sample_frequency=sr, nfft=nfft)
    # Generate output noise signals with the desired spatial coherence.
    signal_noise_coherent, _, _ = anf.generate_signals(signal_noise, params, decomposition='evd', processing='balance+smooth')
    return signal_noise_coherent

