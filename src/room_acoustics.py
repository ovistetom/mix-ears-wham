import numpy as np
import pyroomacoustics as pra
import anf_generator as anf
import random
import itertools 
import torch
import torchaudio
import torchaudio.transforms
import os


SR = 16000
NFFT = 1024


def load_audio_file(file_path):
    signal, sr = torchaudio.load(file_path, channels_first=True)
    if signal.dim() > 1:
        signal = signal[0]
    if sr != SR:
        signal = torchaudio.transforms.Resample(sr, SR)(signal)
    return signal.numpy()


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


def add_noise(signal_mixed, signal_noise, mics_pos, sr=SR, nfft=NFFT):
    # Define target spatial coherence.
    params = anf.CoherenceMatrix.Parameters(mic_positions=mics_pos, sc_type='spherical', sample_frequency=sr, nfft=nfft)
    # Generate output noise signals with the desired spatial coherence.
    signal_noise, _, _ = anf.generate_signals(signal_noise, params, decomposition='evd', processing='balance+smooth')
    if signal_mixed.shape[1] != signal_noise.shape[1]:
        # Determine the maximum size along the second dimension.
        max_size = max(signal_mixed.shape[1], signal_noise.shape[1])
        # Pad both arrays to the maximum size.
        signal_mixed_padded = np.pad(signal_mixed, ((0, 0), (0, max_size - signal_mixed.shape[1])), mode='constant')
        signal_noise_padded = np.pad(signal_noise, ((0, 0), (0, max_size - signal_noise.shape[1])), mode='constant')
        return signal_mixed_padded + signal_noise_padded
    else:
        return signal_mixed + signal_noise

def generate_acoustic_mixture(room_parameters, 
                              signal_clean, 
                              signal_distr, 
                              signal_noise, 
                              distr_snr, 
                              noise_snr, 
                              is_anechoic=False, 
                              root_dir=''):
    # Load room parameters.
    room_dim = room_parameters['room_dim']
    mouth_pos = room_parameters['mouth_pos']
    distr_pos = room_parameters['distr_pos']
    mics_pos = room_parameters['mics_pos']
    e_absorption = room_parameters['e_absorption']
    max_order = room_parameters['max_order']

    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_absorption), max_order=max_order)
    # CASE #1: with distractor, without ambient noise.
    if signal_noise is None:
        # Create room and add necessary sources.
        room.add_source(mouth_pos, signal=signal_clean, delay=0.0)
        room.add_source(distr_pos, signal=signal_distr, delay=0.0)

        # Simulate acoustic scene.
        room.add_microphone_array(mics_pos.T)
        room.simulate()
        signal_mixed = room.mic_array.signals

        # Save audio files.
        signal_mixed = pra.normalize(signal_mixed)
        signal_mixed = torch.from_numpy(signal_mixed).to(torch.float32)
        path_to_mixture_sample = os.path.join(root_dir, f"mixed_distrSNR{distr_snr:+.1f}_noiseSNR+Inf_echo{not is_anechoic}.wav")

    # CASE #2: without distractor, with ambient noise.
    elif signal_distr is None:
        # Create room and add necessary sources.
        room.add_source(mouth_pos, signal=signal_clean, delay=0.0)

        # Simulate acoustic scene.
        room.add_microphone_array(mics_pos.T)
        room.simulate()
        signal_mixed = room.mic_array.signals

        # Add spatially coherent noise.
        signal_mixed = add_noise(signal_mixed, signal_noise, mics_pos)

        # Save audio files.
        signal_mixed = pra.normalize(signal_mixed)
        signal_mixed = torch.from_numpy(signal_mixed).to(torch.float32)
        path_to_mixture_sample = os.path.join(root_dir, f"mixed_distrSNR+Inf_noiseSNR{noise_snr:+.1f}_echo{not is_anechoic}.wav")

    # CASE #3: with distractor, with ambient noise.
    else:
        # Create room and add necessary sources.
        room.add_source(mouth_pos, signal=signal_clean, delay=0.0)
        room.add_source(distr_pos, signal=signal_distr, delay=0.0)

        # Simulate acoustic scene.
        room.add_microphone_array(mics_pos.T)
        room.simulate()
        signal_mixed = room.mic_array.signals

        # Add spatially coherent noise.
        signal_mixed = add_noise(signal_mixed, signal_noise, mics_pos)        

        # Save audio files.
        signal_mixed = pra.normalize(signal_mixed)
        signal_mixed = torch.from_numpy(signal_mixed).to(torch.float32)
        path_to_mixture_sample = os.path.join(root_dir, f"mixed_distrSNR{distr_snr:+.1f}_noiseSNR{noise_snr:+.1f}_echo{not is_anechoic}.wav")
    
    torchaudio.save(path_to_mixture_sample, signal_mixed, SR)


def create_mixture_audio_sample(path_to_speaker_sample, 
                                path_to_distractor_sample, 
                                path_to_noise_sample, 
                                path_to_output_folder='',
                                room_is_anechoic=False, 
                                ):

    # Define acoustic scene.
    room_dim = random_room_dimensions()
    head_pos = random_head_position(room_dim)
    head_ang = random_head_angle()
    ears_pos = random_ears_position(head_pos, head_ang)
    mics_pos = define_mics_position(ears_pos)
    mouth_pos = random_mouth_position(head_pos,  head_ang)
    distr_pos = random_distractor_position(room_dim, head_pos)

    distr_snr = random_snr(-5, 5)
    noise_snr = random_snr(-8, 8)

    # Define acoustic parameters.
    if room_is_anechoic:
        e_absorption, max_order = 1.0, 0
    else:
        rt60 = random_rt60(room_dim)
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room_params = {
        'room_dim': room_dim,
        'head_pos': head_pos,
        'head_ang': head_ang,
        'ears_pos': ears_pos,
        'mics_pos': mics_pos,
        'mouth_pos': mouth_pos,
        'distr_pos': distr_pos,
        'e_absorption': e_absorption,
        'max_order': max_order,
        }
    
    # Load audio files.
    signal_clean = load_audio_file(path_to_speaker_sample)
    signal_distr = load_audio_file(path_to_distractor_sample)
    signal_noise = load_audio_file(path_to_noise_sample)

    # Normalize signals.
    signal_clean = pra.normalize(signal_clean)
    signal_distr = pra.normalize(signal_distr)
    signal_noise = pra.normalize(signal_noise)

    # Apply desired SNR.
    signal_distr = signal_distr * np.power(10, -distr_snr/20)
    signal_noise = signal_noise * np.power(10, -noise_snr/20)

    # Generate three mixtures.
    generate_acoustic_mixture(room_params, signal_clean, signal_distr, signal_noise, distr_snr, noise_snr, is_anechoic=room_is_anechoic, root_dir=path_to_output_folder)
    generate_acoustic_mixture(room_params, signal_clean, None, signal_noise, distr_snr, noise_snr, is_anechoic=room_is_anechoic, root_dir=path_to_output_folder)
    generate_acoustic_mixture(room_params, signal_clean, signal_distr, None, distr_snr, noise_snr, is_anechoic=room_is_anechoic, root_dir=path_to_output_folder)

    # Generate metadata text file.
    with open(os.path.join(path_to_output_folder, "metadata.txt"), 'w') as f:
        f.write(f"Speaker sample:\n\t{path_to_speaker_sample}\n")
        f.write(f"Distractor sample:\n\t{path_to_distractor_sample}\n")
        f.write(f"Noise sample:\n\t{path_to_noise_sample}\n")
        f.write(f"Room dimensions:\n\t{room_dim}\n")
        f.write(f"Head position:\n\t{head_pos}\n")
        f.write(f"Head angle:\n\t{head_ang}\n")
        f.write(f"Ears position:\n\t{ears_pos}\n")
        f.write(f"Mics position:\n\t{mics_pos}\n")
        f.write(f"Mouth position:\n\t{mouth_pos}\n")
        f.write(f"Distractor position:\n\t{distr_pos}\n")


if __name__ == '__main__':

    is_anechoic = False
    # Define audio file paths.
    file_path_clean = os.path.join("database", "VCTK", "wav48_silence_trimmed", "p265", "p265_003_mic1.flac")
    file_path_distr = os.path.join("database", "VCTK", "wav48_silence_trimmed", "p304", "p304_003_mic1.flac")
    file_path_noise = os.path.join("database", "WHAM", "tr", "01aa010b_0.97482_209a010p_-0.97482.wav")
    # Define output path.
    fldr_path_mixed = os.path.join("out", "test")
    os.makedirs(fldr_path_mixed, exist_ok=True)
    # Create mixture.
    create_mixture_audio_sample(file_path_clean, file_path_distr, file_path_noise, room_is_anechoic=is_anechoic, path_to_output_folder=fldr_path_mixed) 

