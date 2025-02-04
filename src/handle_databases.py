import os
import random
import shutil
import torch
import torchaudio
import torchaudio.transforms as tt
import torch.nn.functional as ff

SR = 16000

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


def segment_audio_array(audio_array, segment_length_in_s=20.0, fade_length_in_s=1.0, sample_rate=16000):
    """Segment an audio array into segments of a given length. Apply fade-in and fade-out to the segments.

    Args:
        audio_array (torch.Tensor): The audio array to segment.
        segment_length_in_s (float, optional): The length of each segment in seconds. Defaults to 20.0.
        fade_length_in_s (float, optional): The length of the fade in and out in seconds. Defaults to 1.0.
        sample_rate (int, optional): The sample rate of the audio. Defaults to 16000.

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


def process_vctk(vctk_root):
    """Split the VCTK database into training, validation and test subsets.
    Load all FLAC audio files in VCTK, slice them in segments of length 4s, and save the resulting audio files."""

    vctk_path = os.path.join(vctk_root, 'wav48_silence_trimmed')
    speaker_list = [ d for d in os.listdir(vctk_path) if os.path.isdir(os.path.join(vctk_path, d)) ] 

    def find_start_index(audio, threshold=0.1):
        max_value = audio.max()
        start_index = torch.where(audio > threshold*max_value)[0][0]
        return start_index.item() 
    
    fade_len_in_s = 0.25
    fade_len = int(SR * fade_len_in_s)
    fade_transform = tt.Fade(fade_in_len=fade_len, fade_out_len=fade_len, fade_shape='linear')
    
    num_speakers = len(speaker_list)
    random.shuffle(speaker_list)

    # Compute subset sizes.
    trn_size = round(0.8*num_speakers)
    tst_size = round(0.1*num_speakers)
    val_size = round(0.1*num_speakers)

    # Split shuffled list into subsets.
    trn_subset = speaker_list[:trn_size]
    tst_subset = speaker_list[trn_size:trn_size+tst_size]
    val_subset = speaker_list[-val_size:]

    for subset, subset_name  in [(trn_subset, 'trn'), (tst_subset, 'tst'), (val_subset, 'val')]:
        for speaker in subset:
            speaker_path_src = os.path.join(vctk_path, speaker)
            speaker_path_dst = os.path.join(os.path.dirname(vctk_root), 'sliced_vctk', subset_name, speaker)
            os.makedirs(speaker_path_dst, exist_ok=True)        
            for file_name in os.listdir(speaker_path_src):
                # Load audio file and resample to 16kHz.
                file_path_src = os.path.join(speaker_path_src, file_name)
                audio, sr = torchaudio.load(file_path_src, channels_first=True)
                audio = audio[0]
                resample_transform = tt.Resample(orig_freq=sr, new_freq=SR)
                audio = resample_transform(audio)
                # Find beginning of utterance.
                start_index = find_start_index(audio)
                start_index = max(0, start_index - int(0.25*SR))
                # Slice 4s segment from beginning of utterance.
                audio = audio[start_index:]
                audio = audio[:4*SR] if audio.size(0) > 4*SR else ff.pad(audio, (0, 4*SR-audio.size(0)))
                # Normalize and fade.
                audio /= audio.abs().max()
                audio = fade_transform(audio)
                # Save segment.
                file_path_dst = os.path.join(speaker_path_dst, file_name)
                torchaudio.save(file_path_dst, audio.unsqueeze(0), SR)            

def process_lisp(lisp_root):
    """Split the LibriSpeech database into training, validation and test subsets.
    Load all FLAC audio files in LibriSpeech, slice them in segments of length 4s, and save the resulting audio files."""  

    def find_start_index(audio, threshold=0.1):
        max_value = audio.max()
        start_index = torch.where(audio > threshold*max_value)[0][0]
        return start_index.item()

    fade_len_in_s = 0.25
    fade_len = int(SR * fade_len_in_s)
    fade_transform = tt.Fade(fade_in_len=fade_len, fade_out_len=fade_len, fade_shape='linear')

    for (subset_name, new_subset_name) in [('test-clean', 'tst'), ('train-clean-100', 'trn'), ('dev-clean', 'val')]:
        lisp_path = os.path.join(lisp_root, subset_name)
        speaker_list = [ d for d in os.listdir(lisp_path) if os.path.isdir(os.path.join(lisp_path, d)) ]  
        for speaker_name in speaker_list:
            speaker_path_src = os.path.join(lisp_path, speaker_name)
            speaker_path_dst = os.path.join(os.path.dirname(lisp_root), 'sliced_lisp', new_subset_name, speaker_name)
            os.makedirs(speaker_path_dst, exist_ok=True)            
            for book_name in os.listdir(speaker_path_src):
                book_path_src = os.path.join(speaker_path_src, book_name)
                for file_name in os.listdir(book_path_src):
                    if file_name.endswith('.flac'):
                        file_path_src = os.path.join(book_path_src, file_name)
                    else:
                        continue
                    # Load audio file and resample to 16kHz.
                    audio, sr = torchaudio.load(file_path_src, channels_first=True)
                    audio = audio[0]
                    resample_transform = tt.Resample(orig_freq=sr, new_freq=SR)
                    audio = resample_transform(audio)
                    # Find beginning of utterance.
                    start_index = find_start_index(audio)
                    start_index = max(0, start_index - int(0.25*SR))
                    # Slice 4s segment from beginning of utterance.
                    audio = audio[start_index:]
                    audio = audio[:4*SR] if audio.size(0) > 4*SR else ff.pad(audio, (0, 4*SR-audio.size(0)))
                    # Normalize and fade.
                    audio /= audio.abs().max()
                    audio = fade_transform(audio)
                    # Save segment.
                    file_path_dst = os.path.join(speaker_path_dst, file_name)
                    torchaudio.save(file_path_dst, audio.unsqueeze(0), SR)                 

def process_dmnd(dmnd_root):
    """Split the DEMAND database into training, validation and test subsets.
    Load all WAV audio files in DEMAND, slice them in 4-channel segments of length 4s, and save the resulting audio files."""
    # Generate then shuffle the list of all noise environments.
    list_environments = os.listdir(os.path.join(dmnd_root, '16k'))
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
        for environment in subset:
            envt_name = environment[:-4]
            envt_path_src = os.path.join(dmnd_root, '16k', environment, envt_name)

            envt_path_dst = os.path.join(os.path.dirname(dmnd_root), 'sliced_dmnd', subset_name, envt_name)
            os.makedirs(envt_path_dst, exist_ok=True)

            for file_name in os.listdir(envt_path_src):
                file_path_src = os.path.join(envt_path_src, file_name)
                audio, sr = torchaudio.load(file_path_src, channels_first=True)
                # Slice audio in 16s segments.
                segments_16s = segment_audio_array(audio, segment_length_in_s=16.0, fade_length_in_s=0.0, sample_rate=sr)
                for i, segment_i in enumerate(segments_16s):
                    # Slice segment in 4-channel signals of length 4s.
                    segment_4s = segment_audio_array(segment_i, segment_length_in_s=4.0, fade_length_in_s=0.25, sample_rate=sr)
                    segment_4s = segment_4s.squeeze(1)
                    # Divide each channel by max.
                    segment_4s /= torch.max(segment_4s)
                    # Save 4-channel segment.
                    file_path_dst = os.path.join(envt_path_dst, f'{file_name[:-4]}_{i:02}.flac')
                    torchaudio.save(file_path_dst, segment_4s, sr)


def process_wham(wham_root):
    """Split the WHAM database into training, validation and test subsets.
    Load all audio files in WHAM, slice them in segments of length 4s, and save the resulting audio files."""  

    fade_len_in_s = 0.25
    fade_len = int(SR * fade_len_in_s)
    fade_transform = tt.Fade(fade_in_len=fade_len, fade_out_len=fade_len, fade_shape='linear')
    
    for (subset_name, new_subset_name) in [('tt', 'tst'), ('tr', 'trn'), ('cv', 'val')]:
       
        wham_path_src = os.path.join(wham_root, subset_name)
        wham_list = os.listdir(wham_path_src)

        wham_path_dst = os.path.join(os.path.dirname(wham_root), 'sliced_wham', new_subset_name)
        os.makedirs(wham_path_dst, exist_ok=True) 

        for i, (file_name_1, file_name_2, file_name_3, file_name_4) in enumerate(zip(wham_list[::4], wham_list[1::4], wham_list[2::4], wham_list[3::4])):
            # Load audio files.
            file_path_1 = os.path.join(wham_path_src, file_name_1)
            file_path_2 = os.path.join(wham_path_src, file_name_2)
            file_path_3 = os.path.join(wham_path_src, file_name_3)
            file_path_4 = os.path.join(wham_path_src, file_name_4)
            audio_1, sr_1 = torchaudio.load(file_path_1, channels_first=True)
            audio_2, sr_2 = torchaudio.load(file_path_2, channels_first=True)
            audio_3, sr_3 = torchaudio.load(file_path_3, channels_first=True)
            audio_4, sr_4 = torchaudio.load(file_path_4, channels_first=True)
            assert sr_1 == sr_2 == sr_3 == sr_4 == SR, "Audio files must have the same sample rate."
            # Keep only first channel and first 4s of each audio file (pad if necessary).
            audio_1 = audio_1[0, :4*SR] if audio_1.size(1) > 4*SR else ff.pad(audio_1, (0, 4*SR-audio_1.size(1)), mode='reflect')[0]
            audio_2 = audio_2[0, :4*SR] if audio_2.size(1) > 4*SR else ff.pad(audio_2, (0, 4*SR-audio_2.size(1)), mode='reflect')[0]
            audio_3 = audio_3[0, :4*SR] if audio_3.size(1) > 4*SR else ff.pad(audio_3, (0, 4*SR-audio_3.size(1)), mode='reflect')[0]
            audio_4 = audio_4[0, :4*SR] if audio_4.size(1) > 4*SR else ff.pad(audio_4, (0, 4*SR-audio_4.size(1)), mode='reflect')[0]
            # Create 4-channel segment of length 4s, normalize and fade.
            segment_4s = torch.stack((audio_1, audio_2, audio_3, audio_4), dim=0)
            segment_4s = segment_4s / torch.max(segment_4s, dim=1, keepdim=True)[0]
            segment_4s = fade_transform(segment_4s)
            # Save segment.
            file_path_dst = os.path.join(wham_path_dst, f"{i:05}.flac")
            torchaudio.save(file_path_dst, segment_4s, sr_1)


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


def parse_wham(wham_root, subset='trn'):
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
        environment_path = os.path.join(dmnd_path, environment)
        for file_name in os.listdir(environment_path):
            file_path = os.path.join(environment_path, file_name)
            file_list.append(file_path)
            
    print(f"Found {len(file_list)} files in '{dmnd_path}'.")
    random.shuffle(file_list)
    return file_list

if __name__ == '__main__':

    vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK/VCTK_092"
    lisp_root = r"/home/ovistetom/Documents/Databases_Local/LISP/LibriSpeech"
    dmnd_root = r"/home/ovistetom/Documents/Databases_Local/DMND/DEMAND"
    wham_root = r"/home/ovistetom/Documents/Databases_Local/WHAM/wham_noise"

    # process_vctk(vctk_root)
    process_wham(wham_root)