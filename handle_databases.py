import os
import random
import shutil


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



if __name__ == '__main__':

    vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK_092/wav48_silence_trimmed"