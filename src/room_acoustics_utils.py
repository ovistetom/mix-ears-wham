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


def random_room_dimensions():
    x = random.uniform(4, 10)
    y = random.uniform(4, 10)
    z = random.uniform(2.5, 5)
    return np.array([x, y, z])

def random_head_position(room_dim, rdtw=0.125):
    rdx, rdy, _ = room_dim
    x = random.uniform(rdtw*rdx, (1-rdtw)*rdx)
    y = random.uniform(rdtw*rdy, (1-rdtw)*rdy)
    z = random.uniform(1, 1.75)
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
    mics_pos = np.array([mic_l_dn, mic_l_up, mic_r_dn, mic_r_up])
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

def random_distractor_position(room_dim, head_pos, rdtw=0.125):
    rdx, rdy, _ = room_dim
    hpx, hpy, _ = head_pos
    angle = random.uniform(0, 2*np.pi)

    dx = np.cos(angle)*1
    if dx > 0:
        x_min = hpx + dx
        x_max = (1-rdtw)*rdx
        if x_min > x_max:
            x_min = rdtw*rdx
            x_max = hpx - dx     
    else:
        x_min = rdtw*rdx
        x_max = hpx + dx
        if x_min > x_max:
            x_min = hpx - dx
            x_max = (1-rdtw)*rdx
    x = random.uniform(x_min, x_max)

    dy = np.sin(angle)*1
    if dy > 0:
        y_min = hpy + dy
        y_max = (1-rdtw)*rdy
        if y_min > y_max:
            y_min = rdtw*rdy
            y_max = hpy - dy
    else:
        y_min = rdtw*rdy
        y_max = hpy + dy
        if y_min > y_max:
            y_min = hpy - dy
            y_max = (1-rdtw)*rdy
    y = random.uniform(y_min, y_max)
    
    z = random.uniform(1, 2)
    return np.array([x, y, z])

def random_snr(a=-5, b=5):
    return random.uniform(a, b)

def random_rt60(room_dim):
    v = np.prod(room_dim)
    s = 2 * np.sum([l1 * l2 for l1, l2 in itertools.combinations(room_dim, 2)])
    c = pra.parameters._constants_default['c']
    sc = 24
    a = v*sc*np.log(10)/(c*s)
    return random.uniform(a, 1.0)


def add_noise(signal_mixtr, signal_noise, mics_pos, sr=SR, nfft=NFFT):
    # Define target spatial coherence.
    params = anf.CoherenceMatrix.Parameters(mic_positions=mics_pos, sc_type='spherical', sample_frequency=sr, nfft=nfft)
    # Generate output noise signals with the desired spatial coherence.
    signal_noise_coherent, _, _ = anf.generate_signals(signal_noise, params, decomposition='evd', processing='balance+smooth')
    if signal_mixtr.shape[1] != signal_noise_coherent.shape[1]:
        # Determine the maximum size along the second dimension.
        max_size = max(signal_mixtr.shape[1], signal_noise_coherent.shape[1])
        # Pad both arrays to the maximum size.
        signal_mixtr_padded = np.pad(signal_mixtr, ((0, 0), (0, max_size - signal_mixtr.shape[1])), mode='constant')
        signal_noise_padded = np.pad(signal_noise_coherent, ((0, 0), (0, max_size - signal_noise.shape[1])), mode='constant')
        return signal_mixtr_padded + signal_noise_padded
    else:
        return signal_mixtr + signal_noise_coherent


def generate_acoustic_mixture(
        room_parameters, 
        signal_truth, 
        signal_distr, 
        signal_noise, 
        target_length_in_s=None,
        target_directory='',
):
    # Load room parameters.
    room_dim = room_parameters['room_dim']
    mouth_pos = room_parameters['mouth_pos']
    distr_pos = room_parameters['distr_pos']
    mics_pos = room_parameters['mics_pos']
    e_absorption = room_parameters['e_absorption']
    max_order = room_parameters['max_order']

    # CASE #1: with distractor, without ambient noise.
    if signal_noise is None:
        # Create room and add necessary sources.
        room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_absorption), max_order=max_order)
        room.add_source(mouth_pos, signal=signal_truth, delay=0)
        room.add_source(distr_pos, signal=signal_distr, delay=0)
        # Simulate acoustic scene.
        room.add_microphone_array(mics_pos.T)
        room.simulate()
        sim_signal_mixtr = room.mic_array.signals if room.mic_array is not None else np.array([])
        path_to_mixtr_sample = os.path.join(target_directory, 'mixtr.flac')     

    # CASE #2: without distractor, with ambient noise.
    elif signal_distr is None:
        # Create room and add necessary sources.
        room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_absorption), max_order=max_order)
        room.add_source(mouth_pos, signal=signal_truth, delay=0)
        # Simulate acoustic scene.
        room.add_microphone_array(mics_pos.T)
        room.simulate()
        sim_signal_mixtr = room.mic_array.signals if room.mic_array is not None else np.array([])
        # Add spatially coherent noise.
        sim_signal_mixtr = add_noise(sim_signal_mixtr, signal_noise, mics_pos)
        path_to_mixtr_sample = os.path.join(target_directory, 'mixtr.flac')  

    # CASE #3: with distractor, with ambient noise.
    else:
        # Create room and add necessary sources.
        room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_absorption), max_order=max_order)
        room.add_source(mouth_pos, signal=signal_truth, delay=0)
        room.add_source(distr_pos, signal=signal_distr, delay=0)
        # Simulate acoustic scene.
        room.add_microphone_array(mics_pos.T)
        room.simulate()
        sim_signal_mixtr = room.mic_array.signals if room.mic_array is not None else np.array([])
        # Add spatially coherent noise.
        sim_signal_mixtr = add_noise(sim_signal_mixtr, signal_noise, mics_pos)        
        path_to_mixtr_sample = os.path.join(target_directory, 'mixtr.flac')

    # Repeat to re-create clean speech signal in similar (non-reverberant) simulated conditons.
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(1.0), max_order=0)
    room.add_source(mouth_pos, signal=signal_truth, delay=0)
    room.add_microphone_array(mics_pos.T)
    room.simulate()
    sim_signal_clean = room.mic_array.signals if room.mic_array is not None else np.array([])
    path_to_clean_sample = os.path.join(target_directory, 'clean.flac')
    
    # Slice signal tail if necessary.    
    if target_length_in_s is not None:
        sim_signal_clean = sim_signal_clean[:, :int(SR*target_length_in_s)]
        sim_signal_mixtr = sim_signal_mixtr[:, :int(SR*target_length_in_s)]   

    # Isolate noise (target speech reverberation + distractor speech + ambient noise).
    sim_signal_noise = sim_signal_mixtr - sim_signal_clean
    path_to_noise_sample = os.path.join(target_directory, 'noise.flac')  

    # Normalize mixture signal; save normalization coefficient.
    sim_signal_norm = max(np.abs(sim_signal_clean).max(), np.abs(sim_signal_noise).max(), np.abs(sim_signal_mixtr).max())
    sim_signal_clean /= sim_signal_norm
    sim_signal_noise /= sim_signal_norm
    sim_signal_mixtr /= sim_signal_norm

    # Save signals.        
    sim_signal_clean = torch.from_numpy(sim_signal_clean).to(torch.float32)
    sim_signal_noise = torch.from_numpy(sim_signal_noise).to(torch.float32)
    sim_signal_mixtr = torch.from_numpy(sim_signal_mixtr).to(torch.float32)
    torchaudio.save(path_to_clean_sample, sim_signal_clean, SR)
    torchaudio.save(path_to_noise_sample, sim_signal_noise, SR)
    torchaudio.save(path_to_mixtr_sample, sim_signal_mixtr, SR)
