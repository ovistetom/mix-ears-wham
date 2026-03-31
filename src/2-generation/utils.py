import numpy as np
import pyroomacoustics as pra
import random
import itertools 

# Utility functions for random generation of room and signal parameters.
# Dimensions of room, as well as geometry of sources and microphones, are hardcoded. Modify accordingly if needed.

def random_room_dimensions():
    """Generate random room dimensions (length, width, height) in meters."""
    x = random.randint(4, 10)
    y = random.randint(4, 10)
    z = random.randint(3, 5)
    return np.array([x, y, z])

def random_head_position(room_dim, mdtw=1):
    """Generate random head position (x, y, z) in meters within the given room dimensions."""
    rdx, rdy, _ = room_dim
    # mdtw: minimum distance to the wall.
    x = random.uniform(mdtw, rdx - mdtw)
    y = random.uniform(mdtw, rdy - mdtw)
    z = random.uniform(1, 2)
    return np.array([x, y, z])

def random_head_yaw():
    """Generate random head yaw angle in radians."""
    return random.uniform(0, 2*np.pi)

def random_head_pitch():
    """Generate random head pitch angle in radians."""
    return random.uniform(-0.25*np.pi, 0.25*np.pi)

def random_head_roll():
    """Generate random head roll angle in radians."""
    return random.uniform(-0.125*np.pi, 0.125*np.pi)

def random_mouth_position(head_pos, head_yaw=0.0, head_pitch=0.0, head_roll=0.0):
    """Generate random mouth (i.e. target speaker source) position (x, y, z) in meters based on head position and orientation."""
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
    """Generate random microphone position (x, y, z) in meters based on head position and orientation."""
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
    mics_pos = mics_pos + np.repeat(head_pos[None, ...], repeats=4, axis=0)
    return mics_pos

def random_distractor_position(room_dim, head_pos, mdtw=1):
    """Generate random distractor (i.e. interfering speaker source) position (x, y, z) in meters, at least `mdtw` meters away from head position."""
    rdx, rdy, _ = room_dim
    hpx, hpy, _ = head_pos
    angle = random.uniform(0, 2*np.pi)

    # Generate random x position at the desired distance from head position, making sure to stay within room dimensions.
    dx = np.cos(angle)*1
    if dx > 0:
        x_min = hpx + dx
        x_max = rdx - mdtw
        if x_min > x_max:
            x_min = mdtw
            x_max = hpx - dx     
    else:
        x_min = mdtw
        x_max = hpx + dx
        if x_min > x_max:
            x_min = hpx - dx
            x_max = rdx - mdtw
    x = random.uniform(x_min, x_max)
    # Generate random y position at the desired distance from head position, making sure to stay within room dimensions.
    dy = np.sin(angle)*1
    if dy > 0:
        y_min = hpy + dy
        y_max = rdy - mdtw
        if y_min > y_max:
            y_min = mdtw
            y_max = hpy - dy
    else:
        y_min = mdtw
        y_max = hpy + dy
        if y_min > y_max:
            y_min = hpy - dy
            y_max = rdy - mdtw
    y = random.uniform(y_min, y_max)
    # Generate random z position.
    z = random.uniform(1, 2)
    return np.array([x, y, z])

def random_noise_source_position(room_dim, head_pos, mdtw=0.25, num_sources=16):
    """Generate random noise source position (x, y, z) in meters, at least `mdtw` meters away from head position."""
    rdx, rdy, rdz = room_dim
    hpx, hpy, hpz = head_pos
    list_positions = []
    # Generate random positions until we have the desired number of sources at the desired distance. Quick and dirty.
    while len(list_positions) < num_sources:
        x = random.uniform(mdtw, rdx-mdtw)
        y = random.uniform(mdtw, rdy-mdtw)
        z = random.uniform(mdtw, rdz-mdtw)
        distance = np.sqrt((x-hpx)**2 + (y-hpy)**2 + (z-hpz)**2)
        if distance > 1.0:
            list_positions.append(np.array([x, y, z]))
    return list_positions

def random_snr(a=-5, b=5):
    """Generate random SNR (signal-to-noise ratio) in decibels."""
    return random.uniform(a, b)

def random_rt60(room_dim, max_rt60=1.0):
    """Generate random RT60 (reverberation time) in seconds based on room dimensions. Relies on Sabine's formula."""
    v = np.prod(room_dim)
    s = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
    c = pra.parameters._constants_default['c']
    sc = 24
    a = v*sc*np.log(10)/(c*s)
    return random.uniform(a, max_rt60)

def pad_signal_right(signal, pad_right=0):
    return np.pad(signal, pad_width=((0, 0), (0, pad_right)), mode='constant', constant_values=0.0)

def pad_signal_left_and_right(signal, target_length):
    """Pad signal with zeros on the left and right to reach the target length, with random split of padding on each side."""
    padding_length = target_length - signal.shape[0]
    pad_left = np.random.randint(low=0, high=padding_length)
    pad_right = padding_length - pad_left
    return np.pad(signal, pad_width=(pad_left, pad_right), mode='constant', constant_values=0.0)