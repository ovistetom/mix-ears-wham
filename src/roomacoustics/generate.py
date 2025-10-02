import os
import torch
import torchaudio
import tqdm
import random
import librosa
import numpy as np
import pyroomacoustics as pra
import utils


def generate_acoustic_mixture(
        stem_x: np.ndarray, 
        stem_d: list[np.ndarray], 
        stem_v: np.ndarray, 
        desired_sir: float,
        desired_snr: float,
        room_dim: np.ndarray,
        mouth_pos: np.ndarray,
        distr_pos: list[np.ndarray],
        noise_pos: list[np.ndarray],
        mics_pos: np.ndarray,
        e_absorption: float,
        max_order: int,
        sample_rate: int = utils.SR,
        target_length: int | None = None,
        target_directory: str = '',
):
    # Load room parameters.


    # Create multi-channel dry clean-speech signal.
    room = pra.ShoeBox(room_dim, fs=sample_rate, materials=pra.Material(1.0), max_order=0)
    room.add_source(mouth_pos, signal=stem_x, delay=0)
    room.add_microphone_array(mics_pos.T)
    room.simulate()
    signal_x = room.mic_array.signals if room.mic_array is not None else np.array([])
    power_x = np.pow(signal_x, 2.0).sum()
    path_to_x_sample = os.path.join(target_directory, 'target_speech.flac')

    # Create multi-channel reverberant clean-speech signal.
    room = pra.ShoeBox(room_dim, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(mouth_pos, signal=stem_x, delay=0)
    room.add_microphone_array(mics_pos.T)
    room.simulate()
    signal_r = room.mic_array.signals if room.mic_array is not None else np.array([])
    power_r = np.pow(signal_r, 2.0).sum()

    # Create multi-channel reverberant distractor-speech signal.
    room = pra.ShoeBox(room_dim, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order)
    for distr_pos_i, stem_d_i in zip(distr_pos, stem_d):
        room.add_source(distr_pos_i, signal=stem_d_i, delay=0)
    room.add_microphone_array(mics_pos.T)
    room.simulate()
    signal_d = room.mic_array.signals if room.mic_array is not None else np.array([])
    power_d = np.pow(signal_d, 2.0).sum()
    path_to_d_sample = os.path.join(target_directory, 'interf_speech.flac')
    # Determine SIR.
    initial_sir = power_r / power_d
    signal_d = np.sqrt((initial_sir / desired_sir)).item() * signal_d
    
    # Add spatially coherent noise.
    #signal_v = force_noise_coherence(stem_v, mics_pos=mics_pos)
    # Create multi-channel reverberant distractor-speech signal.
    room = pra.ShoeBox(room_dim, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order)
    for noise_pos_i in noise_pos:
        room.add_source(noise_pos_i, signal=stem_v, delay=0)
    room.add_microphone_array(mics_pos.T)
    room.simulate()
    signal_v = room.mic_array.signals if room.mic_array is not None else np.array([])
    power_v = np.pow(signal_v, 2.0).sum()
    # Determine SNR.
    initial_snr = power_r / power_v
    signal_v = np.sqrt((initial_snr / desired_snr)).item() * signal_v
    path_to_v_sample = os.path.join(target_directory, 'ambient_noise.flac')
    

    # Set target length.
    if target_length is not None:
        signal_v = signal_v[:, :target_length]
        signal_x = signal_x[:, :target_length]
        signal_r = signal_r[:, :target_length]
        signal_d = signal_d[:, :target_length]
    # Speech signals likely to be shorter than noisy mixture: adjust for it.
    signal_x = utils.pad_signal(signal_x, pad_right=signal_v.shape[1] - signal_x.shape[1])
    signal_r = utils.pad_signal(signal_r, pad_right=signal_v.shape[1] - signal_r.shape[1])
    signal_d = utils.pad_signal(signal_d, pad_right=signal_v.shape[1] - signal_d.shape[1])            

    # Create multi-channel noisy mixture signal.
    signal_y = signal_r + signal_d + signal_v
    path_to_y_sample = os.path.join(target_directory, 'noisy_mixture.flac')
    # Define reverb-only signal.
    signal_r = signal_r - signal_x
    path_to_r_sample = os.path.join(target_directory, 'reverb_speech.flac')
    # Define overall-noise signal.
    signal_n = signal_r + signal_d + signal_v
    path_to_n_sample = os.path.join(target_directory, 'overall_noise.flac')

    # Normalize signals w.r.t. maximum; save normalization coefficient.
    signal_max = np.abs(signal_y).max()
    signal_x /= signal_max
    signal_r /= signal_max
    signal_d /= signal_max
    signal_v /= signal_max
    signal_n /= signal_max
    signal_y /= signal_max

    # Save signals.
    torchaudio.save(path_to_x_sample, torch.tensor(signal_x), sample_rate)
    torchaudio.save(path_to_r_sample, torch.tensor(signal_r), sample_rate)
    torchaudio.save(path_to_d_sample, torch.tensor(signal_d), sample_rate)
    torchaudio.save(path_to_v_sample, torch.tensor(signal_v), sample_rate)
    torchaudio.save(path_to_n_sample, torch.tensor(signal_n), sample_rate)
    torchaudio.save(path_to_y_sample, torch.tensor(signal_y), sample_rate)
    return signal_max


def create_acoustic_scene(
    path_to_stem_x: str,
    path_to_stem_d: list[str],
    path_to_stem_v: str,
    target_directory: str,
):
    
    room_dim = utils.random_room_dimensions()
    head_pos = utils.random_head_position(room_dim)
    head_yaw = utils.random_head_yaw()
    head_pitch = utils.random_head_pitch()
    head_roll = utils.random_head_roll()
    mics_pos = utils.random_mics_position(head_pos, head_yaw, head_pitch, head_roll)
    mouth_pos = utils.random_mouth_position(head_pos, head_yaw, head_pitch, head_roll)
    distr_pos = [utils.random_distractor_position(room_dim, head_pos) for _ in path_to_stem_d]
    noise_pos = utils.random_noise_source_position(room_dim, head_pos,num_sources=48)
    desired_sir = utils.random_snr(-6, 6)
    desired_sir = 10**(0.1*desired_sir)
    desired_snr = utils.random_snr(-6, 6)
    desired_snr = 10**(0.1*desired_snr)
    rt60 = utils.random_rt60(room_dim)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    stem_v = torchaudio.load(path_to_stem_v, channels_first=True)[0].squeeze(0).numpy()
    stem_x = utils.pad_and_adjust(
        signal = torchaudio.load(path_to_stem_x, channels_first=True)[0].squeeze(0).numpy(),
        target_length = stem_v.shape[0],
    )
    stem_d = [utils.pad_and_adjust(
        signal = torchaudio.load(p, channels_first=True)[0].squeeze(0).numpy(),
        target_length = stem_v.shape[0],
        ) for p in path_to_stem_d
    ]

    generate_acoustic_mixture(
        stem_x=stem_x,
        stem_d=stem_d,
        stem_v=stem_v,
        room_dim=room_dim,
        mics_pos=mics_pos,
        mouth_pos=mouth_pos,
        distr_pos=distr_pos,
        noise_pos=noise_pos,
        desired_sir=desired_sir,
        desired_snr=desired_snr,
        e_absorption=e_absorption,
        max_order=6, #max_order,
        target_length=stem_v.shape[0],
        target_directory=target_directory,
    )


def parse_database(path_database):
    elements_database = []
    for name_sample in sorted(os.listdir(path_database)):
        path_sample = os.path.join(path_database, name_sample)
        elements_database.append((path_sample, librosa.get_duration(path=path_sample)))
    return elements_database


if __name__ == '__main__':

    subsets = ['trn', 'tst', 'val']
    path_speech = r"/home/ovistetom/Documents/Databases_Local/EARS/preprocessed"
    #path_lisp = r"/home/ovistetom/Documents/Databases_Local/EARS/preprocessed"
    path_noise = r"/home/ovistetom/Documents/Databases_Local/WHAM/preprocessed"
    path_output = r"/home/ovistetom/Documents/Databases_Local/MIXTURES/reference"
    
    for subset in subsets:

        subset_speech = os.path.join(path_speech, subset)
        elements_speech = parse_database(subset_speech)
        subset_noise = os.path.join(path_noise, subset)
        elements_noise = parse_database(subset_noise)
        subset_output = os.path.join(path_output, subset)
        os.makedirs(subset_output, exist_ok=True)

        num_sample = 0

        for path_sample_speech, len_sample_speech in tqdm.tqdm(elements_speech):

            if len_sample_speech < 6:
                continue

            path_sample_noise, len_sample_noise = random.sample(
                population = [x for x in elements_noise if (x[1]>len_sample_speech and x[1]<len_sample_speech+3)], 
                k = 1,
            )[0]
            
            path_sample_distr = random.sample(
                population = [x[0] for x in elements_speech if (x[1]<len_sample_noise and x[0]!=path_sample_speech)], 
                k = 4,
            )

            path_sample = os.path.join(subset_output, f"{num_sample:05d}")
            os.makedirs(path_sample, exist_ok=True)

            create_acoustic_scene(
                path_to_stem_x=path_sample_speech,
                path_to_stem_d=path_sample_distr,
                path_to_stem_v=path_sample_noise,
                target_directory=path_sample,
            )

            num_sample += 1

            #if num_sample == 100:
            #    break

            