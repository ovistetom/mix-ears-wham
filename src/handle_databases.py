import os
import random
import shutil
import torch
import torchaudio
import torchaudio.transforms as tt


def split_vctk(vctk_root):
    """Split VCTK databases into training, validation and test subsets."""
    # Generate then shuffle the list of all VCTK speakers.
    list_speakers = [d for d in os.listdir(vctk_root) if os.path.isdir(os.path.join(vctk_root, d))]
    num_speakers = len(list_speakers)
    random.shuffle(list_speakers)

    # Compute subset sizes.
    trn_size = round(0.8*num_speakers)
    tst_size = round(0.1*num_speakers)
    val_size = round(0.1*num_speakers)

    # Split shuffled list into subsets.
    trn_subset = list_speakers[:trn_size]
    tst_subset = list_speakers[trn_size:trn_size+tst_size]
    val_subset = list_speakers[-val_size:]

    # Move speakers directories to corresponding subset directory.
    for subset, subset_name  in [(trn_subset, 'trn'), (tst_subset, 'tst'), (val_subset, 'val')]:
        dst_path = os.path.join(vctk_root, subset_name)
        os.makedirs(dst_path, exist_ok=True)
        for speaker in subset:
            src_path = os.path.join(vctk_root, speaker)
            shutil.move(src_path, dst_path)


def split_lisp(lisp_root):
    """Split LibriSpeech 'train-clean-100' subset into 'test-clean' and 'dev-clean'."""
    # Generate then shuffle the list of all VCTK speakers.
    train_root = os.path.join(lisp_root, 'train-clean-100')
    list_speakers = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    num_speakers = len(list_speakers)
    random.shuffle(list_speakers)

    # Compute test subset size.
    tst_size = round(0.5*num_speakers)

    # Split shuffled list into subsets.
    tst_subset = list_speakers[:tst_size]
    dev_subset = list_speakers[tst_size:]

    # Move speakers directories to corresponding subset directory.
    for subset, subset_name  in [(tst_subset, 'test-clean'), (dev_subset, 'dev-clean')]:
        dst_path = os.path.join(lisp_root, subset_name)
        os.makedirs(dst_path, exist_ok=True)
        for speaker in subset:
            src_path = os.path.join(train_root, speaker)
            shutil.move(src_path, dst_path)

def split_demand(demand_root):
    """Split the DEMAND database into training, validation and test subsets."""
    # Generate then shuffle the list of all noise types.
    list_environments = [d for d in os.listdir(demand_root) if d.endswith('16k')]
    num_environments = len(list_environments)
    random.shuffle(list_environments)

    # Compute subset sizes.
    trn_size = round(0.8*num_environments)
    tst_size = round(0.1*num_environments)
    val_size = round(0.1*num_environments)

    # Split shuffled list into subsets.
    trn_subset = list_environments[:trn_size]
    tst_subset = list_environments[trn_size:trn_size+tst_size]
    val_subset = list_environments[-val_size:]    


    # Move environment directories to corresponding subset directory.
    for subset, subset_name  in [(trn_subset, 'trn'), (tst_subset, 'tst'), (val_subset, 'val')]:
        dst_path = os.path.join(demand_root, subset_name)
        os.makedirs(dst_path, exist_ok=True)
        for speaker in subset:
            src_path = os.path.join(demand_root, speaker)
            shutil.move(src_path, dst_path)  


def segment_audio_array(audio_array, segment_length_in_s=20.0, fade_length_in_s=1.0, sample_rate=44100):
    """Segment an audio array into segments of a given length. Apply fade-in and fade-out to the segments.

    Args:
        audio_array (torch.Tensor): The audio array to segment.
        segment_length_in_s (float, optional): The length of each segment in seconds. Defaults to 20.0.
        fade_length_in_s (float, optional): The length of the fade in and out in seconds. Defaults to 1.0.
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.

    Returns:
        segments (torch.Tensor): The segmented audio array.
    """
    assert audio_array.dim() == 2, f"Expected 2D tensor, got {audio_array.dim()}D array."

    # Segment the audio array.
    num_channels, audio_length = audio_array.shape
    segment_length = int(sample_rate * segment_length_in_s)
    num_segments = audio_length // segment_length

    segments = torch.zeros((num_segments, num_channels, segment_length))

    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segments[i, :, :] = audio_array[:, start:end]

    # Apply fade-in and fade-out.
    fade_length = int(sample_rate * fade_length_in_s)
    fade_transform = tt.Fade(fade_in_len=fade_length, fade_out_len=fade_length, fade_shape='linear')
    faded_segments = fade_transform(segments)

    return faded_segments


def slice_demand(demand_root, subset='trn'):
    """Load all WAV audio files in `demand_root`, slice them in segments of length 10s and save the resulting audio files."""
    demand_root = os.path.join(demand_root, subset)
    for environment in os.listdir(demand_root):
        environment_path = os.path.join(demand_root, environment, environment[:-4])
        for file_name in os.listdir(environment_path):
            file_path_src = os.path.join(environment_path, file_name)
            audio, sr = torchaudio.load(file_path_src, channels_first=True)
            faded_segments = segment_audio_array(audio, segment_length_in_s=5.0, fade_length_in_s=0.5, sample_rate=sr)
            for i, segment_i in enumerate(faded_segments):
                slices_path = os.path.join(demand_root, environment, 'SLICES')
                os.makedirs(slices_path, exist_ok=True)
                file_path_dst = os.path.join(slices_path, f'{file_name[:-4]}_{i:03}.wav')
                torchaudio.save(file_path_dst, segment_i, sr)


def parse_vctk(vctk_root, subset='trn'):
    """Parse database and return shuffled lists (one for 'mic1', one for 'mic2') of all files in the given subset."""
    vctk_path = os.path.join(vctk_root, subset)
    mic1_list = []
    mic2_list = []

    for speaker_name in os.listdir(vctk_path):
        speaker_path = os.path.join(vctk_path, speaker_name)
        for file_name in os.listdir(speaker_path):

            utterance_name, flac = os.path.splitext(file_name)
            utterance_name = utterance_name[:-4]
            mic1_path = os.path.join(speaker_path, f'{utterance_name}mic1{flac}')
            mic2_path = os.path.join(speaker_path, f'{utterance_name}mic2{flac}')

            if 'mic2' in file_name:
                mic2_list.append(mic2_path)
            elif os.path.exists(mic2_path):
                mic1_list.append(mic1_path)
            else:
                mic1_list.append(mic1_path)
                mic2_list.append(mic1_path)
            
    print(f"Found {len(mic1_list)} (x2) files in '{vctk_path}'.")
    random.shuffle(mic1_list)
    random.shuffle(mic2_list)
    return mic1_list, mic2_list


def parse_wham(wham_root, subset='tr'):
    """Parse database and return shuffled list of all files in the given subset."""
    wham_path = os.path.join(wham_root, subset)
    file_list = []

    for file_name in os.listdir(wham_path):
        file_path = os.path.join(wham_path, file_name)
        file_list.append(file_path)
            
    print(f"Found {len(file_list)} files in '{wham_path}'.")
    random.shuffle(file_list)
    return file_list


def parse_lisp(lisp_root, subset='train-clean-100'):
    """Parse database and return shuffled list of all files in the given subset."""
    lisp_path = os.path.join(lisp_root, subset)
    file_list = []

    for speaker_name in os.listdir(lisp_path):
        speaker_path = os.path.join(lisp_path, speaker_name)
        for book_name in os.listdir(speaker_path):
            book_path = os.path.join(speaker_path, book_name)
            for file_name in os.listdir(book_path):
                file_path = os.path.join(book_path, file_name)
                file_ext = os.path.splitext(file_path)[1]
                if file_ext == '.flac':
                    file_list.append(file_path)
            
    print(f"Found {len(file_list)} files in '{lisp_path}'.")
    random.shuffle(file_list)
    return file_list


def parse_dmnd(dmnd_root, subset='trn'):
    """Parse DEMAND database and return shuffled list of all files in the given subset."""
    dmnd_path = os.path.join(dmnd_root, subset)
    file_list = []

    for environment in os.listdir(dmnd_path):
        environment_path = os.path.join(dmnd_path, environment, 'SLICES')
        for file_name in os.listdir(environment_path):
            file_path = os.path.join(environment_path, file_name)
            file_list.append(file_path)
            
    print(f"Found {len(file_list)} files in '{dmnd_path}'.")
    random.shuffle(file_list)
    return file_list

if __name__ == '__main__':

    vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK_092/wav48_silence_trimmed"
    wham_root = r"/home/ovistetom/Documents/Databases_Local/LIBRIMIX/wham_noise"
    lisp_root = r"/home/ovistetom/Documents/Databases_Local/LIBRIMIX/LibriSpeech"
    dmnd_root = r"/home/ovistetom/Documents/Databases_Local/DEMAND/16k"

    vctk_list = parse_vctk(vctk_root, subset='tst')
    wham_list = parse_wham(wham_root, subset='tt')
    lisp_list = parse_lisp(lisp_root, subset='test-clean')
    dmnd_list = parse_dmnd(dmnd_root, subset='tst')