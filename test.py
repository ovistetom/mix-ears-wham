import numpy as np
import pyroomacoustics as pra
import random
import itertools 
import torch
import torchaudio
import torchaudio.transforms


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

def random_head_angle():
    return random.uniform(0, 2*np.pi)

def random_mouth_position(head_pos, head_ang):
    hpx, hpy, hpz = head_pos
    rdx = random.uniform(-0.01, 0.01)
    rdy = random.uniform(0.11, 0.15)
    rdz = random.uniform(-0.04, -0.02)
    x = hpx + rdx*np.cos(head_ang) - rdy*np.sin(head_ang)
    y = hpy + rdx*np.sin(head_ang) + rdy*np.cos(head_ang)
    z = hpz + rdz
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
    
    z = random.uniform(1, 1.75)
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

def random_diffuse_noise_position(room_dim, num_sources=12):
    rdx, rdy, rdz = room_dim
    return [np.array([random.uniform(0, rdx), random.uniform(0, rdy), random.uniform(0, rdz)]) for _ in range(num_sources)]

def load_audio_file(file_path):
    signal, sr = torchaudio.load(file_path, channels_first=True)
    if signal.dim() > 1:
        signal = signal[0]
    if sr != SR:
        signal = torchaudio.transforms.Resample(sr, SR)(signal)
    return signal.numpy()


if __name__ == '__main__':

    SR = 16000
    is_anechoic = False

    # Create acoustic scene.
    room_dim = random_room_dimensions()
    head_pos = random_head_position(room_dim)
    head_ang = random_head_angle()
    ears_pos = random_ears_position(head_pos, head_ang)
    mics_pos = define_mics_position(ears_pos)
    mouth_pos = random_mouth_position(head_pos,  head_ang)
    distr_pos = random_distractor_position(room_dim, head_pos)
    noise_pos = random_diffuse_noise_position(room_dim, num_sources=16)

    distr_snr = random_snr(-5, 5)
    noise_snr = random_snr(-8, 8)

    # Create shoebox room.
    if is_anechoic:
        e_absorption, max_order = 1.0, 0
    else:
        rt60 = random_rt60(room_dim)
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_absorption), max_order=max_order)

    # Load audio files.
    file_path_clean = r"database\VCTK\wav48_silence_trimmed\p225\p225_004_mic1.flac"
    signal_clean = load_audio_file(file_path_clean)
    file_path_distr = r"database\VCTK\wav48_silence_trimmed\p304\p304_004_mic1.flac"
    signal_distr = load_audio_file(file_path_distr)
    file_path_noise = r"database\WHAM\tr\01aa010b_0.97482_209a010p_-0.97482.wav"    
    signal_noise = load_audio_file(file_path_noise)

    # Normalize signals.
    signal_clean = pra.normalize(signal_clean)
    signal_distr = pra.normalize(signal_distr)
    signal_noise = pra.normalize(signal_noise)

    # Apply desired SNR.
    signal_distr = signal_distr * np.power(10, -distr_snr/20)
    signal_noise = signal_noise * np.power(10, -noise_snr/20)

    # Add sources to acoustic scene.
    room.add_source(mouth_pos, signal=signal_clean, delay=0.0)
    room.add_source(distr_pos, signal=signal_distr, delay=0.0)
    for pos in noise_pos:
        room.add_source(pos, signal=signal_noise, delay=0.0)
    
    # Simulate acoustic scene.
    room.add_microphone_array(mics_pos.T)
    room.simulate()
    signal_mixed = room.mic_array.signals
    signal_mixed = pra.normalize(signal_mixed)

    # Save audio files.
    signal_mixed = torch.from_numpy(signal_mixed).to(torch.float32)
    torchaudio.save(f"out/mixed_distrSNR{distr_snr:+.1f}_noiseSNR{noise_snr:+.1f}_echo{not is_anechoic}.wav", signal_mixed, SR)

    # TODO: Create acoustic scene without distractor, without noise, without echo.
    # IDEA: Use VCTK mic1 with echo, use VTCK mic2 without echo... no strong reason, but it allows to use the full VTCK DB, adding a tiny bit more diversity.
    # TODO: Create metadata file with various parameters: room size, positioning, etc.
    # TODO: Add to metadata the speaker and utterance numbers from VCTK, LibriSpeech and WHAM. 
