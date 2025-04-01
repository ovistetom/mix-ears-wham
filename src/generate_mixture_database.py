import os
import numpy as np
import torch
import torchaudio
import pyroomacoustics as pra
from tqdm import tqdm
import room_acoustics_utils as utils
from preprocess_databases import parse_vctk, parse_lisp, parse_dmnd


SR = 16000


def load_audio_file_and_resample(file_path, new_sr=SR):
    signal, sr = torchaudio.load(file_path, channels_first=True)
    if sr != new_sr:
        signal = torchaudio.transforms.Resample(sr, new_sr)(signal)
    return signal.numpy()


def define_mixtr_file_name_suffix(
        path_to_speaker_sample,
        path_to_distractor_sample,
        path_to_noise_sample,
        distr_snr,
        noise_snr,
        room_is_anechoic,
):
    # Extract speaker ID.
    speaker_id = path_to_speaker_sample.split('/')[-2].upper()

    # Handle absence of distractor or noise.
    if noise_snr is None:
        str_noise_snr = "+Inf"
        noise_type = "NONE"
    else:
        str_noise_snr = f"{round(noise_snr):+}"
        noise_type = path_to_noise_sample.split('/')[-2].upper()
    if distr_snr is None:
        str_distr_snr = "+Inf"
        distractor_id = "NONE"
    else:
        str_distr_snr = f"{round(distr_snr):+}"
        distractor_id = path_to_distractor_sample.split('/')[-2].upper()

    return f"speaker{speaker_id}_distractor{distractor_id}_noiseType{noise_type}_distrSNR{str_distr_snr}_noiseSNR{str_noise_snr}_echo{not room_is_anechoic}"


def create_three_mixture_audio_samples(
        path_to_speaker_sample, 
        path_to_distractor_sample, 
        path_to_noise_sample, 
        path_to_output_folder='',
        output_counter=0,
        room_is_anechoic=False,
        target_length_in_s=4.0,
        normalize_signals=False,
):
    # Define acoustic scene.
    room_dim = utils.random_room_dimensions()
    head_pos = utils.random_head_position(room_dim)
    head_yaw = utils.random_head_yaw()
    head_pitch = utils.random_head_pitch()
    head_roll = utils.random_head_roll()
    mics_pos = utils.random_mics_position(head_pos, head_yaw, head_pitch, head_roll)
    mouth_pos = utils.random_mouth_position(head_pos, head_yaw, head_pitch, head_roll)
    distr_pos = utils.random_distractor_position(room_dim, head_pos)
    distr_snr = utils.random_snr(-15, 0)
    noise_snr = utils.random_snr(-30, 0)

    # Define acoustic parameters.
    if room_is_anechoic:
        e_absorption, max_order = 1.0, 0
    else:
        rt60 = utils.random_rt60(room_dim)
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room_params = {
        'room_dim': room_dim,
        'head_pos': head_pos,
        'mics_pos': mics_pos,
        'mouth_pos': mouth_pos,
        'distr_pos': distr_pos,
        'e_absorption': e_absorption,
        'max_order': max_order,
        }
    
    # Load audio files (keep only first channel for speech signals).
    signal_truth = load_audio_file_and_resample(path_to_speaker_sample)[0]
    signal_distr = load_audio_file_and_resample(path_to_distractor_sample)[0]
    signal_noise = load_audio_file_and_resample(path_to_noise_sample)    

    # Normalize signals.
    if normalize_signals:
        signal_truth /= np.abs(signal_truth).max()
        signal_distr /= np.abs(signal_distr).max()
        signal_noise /= np.abs(signal_noise).max()

    # Apply target SNR.
    power_clean = np.pow(signal_truth, 2).mean()
    power_distr = np.pow(signal_distr, 2).mean()
    power_noise = np.pow(signal_noise, 2).mean()
    distr_current_snr = 10*np.log10(power_clean/power_distr)
    noise_current_snr = 10*np.log10(power_clean/power_noise)
    signal_distr *= 10**((distr_current_snr - distr_snr)/20)
    signal_noise *= 10**((noise_current_snr - noise_snr)/20)

    # Create output folder.
    output_folder_c = os.path.join(path_to_output_folder, f'{output_counter:06}')
    os.makedirs(output_folder_c, exist_ok=True)
    # Generate first mixture.
    utils.generate_acoustic_mixture(
        room_params, 
        signal_truth, 
        signal_distr, 
        signal_noise, 
        target_directory = output_folder_c,
        target_length_in_s = target_length_in_s,
        filename_suffix = define_mixtr_file_name_suffix(
            path_to_speaker_sample,
            path_to_distractor_sample,
            path_to_noise_sample,
            distr_snr,
            noise_snr,
            room_is_anechoic,
        ),
    )
    # Save clean signal.
    torchaudio.save(os.path.join(output_folder_c, 'truth.flac'), torch.from_numpy(signal_truth).to(torch.float32).unsqueeze(0), SR)

    # Create output folder.
    output_folder_c = os.path.join(path_to_output_folder, f'{output_counter+1:06}')
    os.makedirs(output_folder_c, exist_ok=True)    
    # Generate second mixture (without distractor).
    utils.generate_acoustic_mixture(
        room_params, 
        signal_truth, 
        None, 
        signal_noise, 
        target_directory = output_folder_c, 
        target_length_in_s = target_length_in_s,
        filename_suffix = define_mixtr_file_name_suffix(
            path_to_speaker_sample,
            path_to_distractor_sample,
            path_to_noise_sample,
            None,
            noise_snr,
            room_is_anechoic,
        ),
    )
    # Save clean signal.
    torchaudio.save(os.path.join(output_folder_c, 'truth.flac'), torch.from_numpy(signal_truth).to(torch.float32).unsqueeze(0), SR)

    # Create output folder.
    output_folder_c = os.path.join(path_to_output_folder, f'{output_counter+2:06}')
    os.makedirs(output_folder_c, exist_ok=True)
    # Generate second mixture (without background noise).
    utils.generate_acoustic_mixture(
        room_params, 
        signal_truth, 
        signal_distr, 
        None, 
        target_directory = output_folder_c, 
        target_length_in_s = target_length_in_s,
        filename_suffix = define_mixtr_file_name_suffix(
            path_to_speaker_sample,
            path_to_distractor_sample,
            path_to_noise_sample,
            distr_snr,
            None,
            room_is_anechoic,
        ),
    )
    # Save clean signal.
    torchaudio.save(os.path.join(output_folder_c, 'truth.flac'), torch.from_numpy(signal_truth).to(torch.float32).unsqueeze(0), SR)

    # # Create metadata text file.
    # with open(os.path.join(path_to_output_folder, 'metadata.txt'), 'w') as f:
    #     f.write(f"Speaker sample:\n\t{path_to_speaker_sample}\n")
    #     f.write(f"Distractor sample:\n\t{path_to_distractor_sample}\n")
    #     f.write(f"Noise sample:\n\t{path_to_noise_sample}\n")
    #     f.write(f"Distractor SNR:\n\t{distr_snr}\n")
    #     f.write(f"Noise SNR:\n\t{noise_snr}\n")
    #     f.write(f"Room dimensions:\n\t{room_dim}\n")
    #     f.write(f"Head position:\n\t{head_pos}\n")
    #     f.write(f"Mics position:\n\t{mics_pos}\n")
    #     f.write(f"Mouth position:\n\t{mouth_pos}\n")
    #     f.write(f"Distractor position:\n\t{distr_pos}\n")


if __name__ == '__main__':
    
    subsets = ['trn', 'tst', 'val']
    vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK/sliced_vctk"
    lisp_root = r"/home/ovistetom/Documents/Databases_Local/LISP/sliced_lisp"
    dmnd_root = r"/home/ovistetom/Documents/Databases_Local/DMND/sliced_dmnd"
    out_root = os.path.join(r"/home/ovistetom/Documents/Databases_Local/MIXTURES")

    for subset in subsets:
        # Load speech and noise databases.
        vctk_list = parse_vctk(vctk_root, subset=subset)
        lisp_list = parse_lisp(lisp_root, subset=subset)
        dmnd_list = parse_dmnd(dmnd_root, subset=subset)
        path_to_subset_folder = os.path.join(out_root, subset)
        c = 0

        for (file_path_truth, file_path_distr, file_path_noise) in tqdm(zip(vctk_list, lisp_list, dmnd_list), total=len(vctk_list), desc=f"Processing subset '{subset}'"):


            # Define output path.
            # os.makedirs(os.path.join(out_i, 'echoTrue'), exist_ok=True)
            # os.makedirs(os.path.join(out_i, 'echoFalse'), exist_ok=True)

            # Create mixture.
            create_three_mixture_audio_samples(file_path_truth,
                                               file_path_distr, 
                                               file_path_noise, 
                                               room_is_anechoic=False, 
                                               target_length_in_s=4.0,
                                               path_to_output_folder=path_to_subset_folder,
                                               output_counter=c,
            )
            c = c + 3

            create_three_mixture_audio_samples(file_path_truth,
                                               file_path_distr, 
                                               file_path_noise, 
                                               room_is_anechoic=True, 
                                               target_length_in_s=4.0,
                                               path_to_output_folder=path_to_subset_folder,
                                               output_counter=c,
            )
            c = c + 3

