import os
import random
import shutil
import torch
import torchaudio
import torchaudio.transforms as tt
import torch.nn.functional as ff


SR = 16000


def rearrange_lisp_subsets(lisp_root):
    """Split LibriSpeech 'train-clean-100' subset into 'test-clean' and 'dev-clean'."""
    # Generate and shuffle the list of all LibriSpeech speakers in 'train-clean-100'.
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
    """Segment an audio array into segments of a given length. Apply fade-in and fade-out to the segments."""
    # Initialize segments array.
    audio_length = audio_array.size(0)
    segment_length = int(sample_rate * segment_length_in_s)
    num_segments = audio_length // segment_length
    segments = torch.zeros((num_segments, segment_length))
    # Fill segments array.
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segments[i, :] = audio_array[start:end]
    # Apply fade-in and fade-out.
    fade_length = int(sample_rate * fade_length_in_s)
    fader = tt.Fade(fade_in_len=fade_length, fade_out_len=fade_length, fade_shape='linear')
    faded_segments = fader(segments)
    return faded_segments


def find_first_last_index(audio, threshold=0.05):
    """Find the first and last indices in an audio tensor when 0.05*maximum is reached."""
    first_index = torch.where(audio > threshold*audio.max())[0][0]
    last_index = torch.where(audio > threshold*audio.max())[0][-1]
    return first_index.item(), last_index.item()


def center_audio_tensor(audio_tensor):
    return (audio_tensor - audio_tensor.mean())


def pad_end_of_audio_segment(audio_tensor, sr, target_len_in_s=4.0, fade_out_len_in_s=None, padding_mode='constant'):
    """Pad the end of an audio segment to a given length. Apply fade-out to the segment if necessary."""
    audio_length = audio_tensor.size(0)
    target_length = int(sr*target_len_in_s)

    if fade_out_len_in_s is not None:
        fade_length = int(sr*fade_out_len_in_s)
        fader = tt.Fade(fade_in_len=0, fade_out_len=fade_length, fade_shape='linear')
        audio_tensor = fader(audio_tensor) 

    if audio_length > target_length:
        return audio_tensor[:target_length]
    else:
        audio_padded = ff.pad(audio_tensor.unsqueeze(0), (0, target_length-audio_length), mode=padding_mode)[0]
        return audio_padded
    

def pad_beginning_of_audio_segment(audio_tensor, sr, target_len_in_s=4.0, fade_in_len_in_s=None, padding_mode='constant'):
    """Pad the beginning of an audio segment to a given length. Apply fade-in to the segment if necessary."""
    audio_length = audio_tensor.size(0)
    target_length = int(sr*target_len_in_s)

    if fade_in_len_in_s is not None:
        fade_length = int(sr*fade_in_len_in_s)
        fader = tt.Fade(fade_in_len=fade_length, fade_out_len=0, fade_shape='linear')
        audio_tensor = fader(audio_tensor)

    if audio_length > target_length:
        return audio_tensor[-target_length:]
    else:
        audio_padded = ff.pad(audio_tensor.unsqueeze(0), (target_length-audio_length, 0), mode=padding_mode)[0]
        return audio_padded


def pad_and_slice_audio_segment(audio_tensor, sr, target_len_in_s=4.0, fade_in_len_in_s=None, fade_out_len_in_s=None, padding_mode='constant'):
    """Pad and slice an audio segment to a given length. Length of padding is chosen randomly."""
    audio_len_in_s = audio_tensor.size(0)/sr
    padded_len_in_s = random.uniform(audio_len_in_s, target_len_in_s)
    audio_tensor = pad_beginning_of_audio_segment(audio_tensor, sr, padded_len_in_s, fade_in_len_in_s=fade_in_len_in_s, padding_mode=padding_mode)
    audio_tensor = pad_end_of_audio_segment(audio_tensor, sr, target_len_in_s, fade_out_len_in_s=fade_out_len_in_s, padding_mode=padding_mode)
    return audio_tensor


def process_vctk(vctk_root):
    """Split the VCTK database into training, validation and test subsets.
    Load all FLAC audio files in VCTK, slice them in segments of length 4s, and save the resulting audio files."""

    # Generate and shuffle the list of all VCTK speakers.
    vctk_path = os.path.join(vctk_root, 'wav48_silence_trimmed')
    speaker_list = [ d for d in os.listdir(vctk_path) if os.path.isdir(os.path.join(vctk_path, d)) ] 
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

    # Parse and process each subset.
    for subset, subset_name  in [(trn_subset, 'trn'), (tst_subset, 'tst'), (val_subset, 'val')]:
        for speaker in subset:
            speaker_path_src = os.path.join(vctk_path, speaker)
            speaker_path_dst = os.path.join(os.path.dirname(vctk_root), 'sliced_vctk', subset_name, speaker)
            os.makedirs(speaker_path_dst, exist_ok=True)        
            for file_name in os.listdir(speaker_path_src):
                # Load audio file (keep only first channel).
                file_path_src = os.path.join(speaker_path_src, file_name)
                audio, sr = torchaudio.load(file_path_src, channels_first=True)
                audio = audio[0]
                # Resample to 16kHz and center around zero.
                resampler = tt.Resample(orig_freq=sr, new_freq=SR)
                audio = resampler(audio)
                audio = center_audio_tensor(audio)
                # Find beginning and end of utterance.
                first_index, last_index = find_first_last_index(audio)
                first_index = max(0, first_index - int(SR*random.uniform(0.2, 0.5)))
                last_index = min(audio.size(0), last_index + int(SR*random.uniform(0.2, 0.5)))
                audio = audio[first_index:last_index]
                # Pad the audio segment to 4s.
                audio = pad_and_slice_audio_segment(audio, SR, target_len_in_s=4.0, fade_in_len_in_s=random.uniform(0.1, 0.4), fade_out_len_in_s=random.uniform(0.1, 0.4))
                # Fade and normalize.
                fader = tt.Fade(fade_in_len=1024, fade_out_len=1024, fade_shape='linear')                
                audio = fader(audio)
                audio /= audio.abs().max()
                # Save segment.
                file_path_dst = os.path.join(speaker_path_dst, file_name)
                torchaudio.save(file_path_dst, audio.unsqueeze(0), SR)            


def process_lisp(lisp_root):
    """Split the LibriSpeech database into training, validation and test subsets.
    Load all FLAC audio files in LibriSpeech, slice them in segments of length 4s, and save the resulting audio files."""  

    # Parse and process each subset.
    for (subset_name, new_subset_name) in [('test-clean', 'tst'), ('train-clean-360', 'trn'), ('dev-clean', 'val')]:
        lisp_path = os.path.join(lisp_root, subset_name)
        speaker_list = [ d for d in os.listdir(lisp_path) if os.path.isdir(os.path.join(lisp_path, d)) ]  
        for speaker_name in speaker_list:
            speaker_path_src = os.path.join(lisp_path, speaker_name)
            speaker_path_dst = os.path.join(os.path.dirname(lisp_root), 'sliced_lisp', new_subset_name, speaker_name)
            os.makedirs(speaker_path_dst, exist_ok=True)            
            for book_name in os.listdir(speaker_path_src):
                book_path_src = os.path.join(speaker_path_src, book_name)
                for file_name in os.listdir(book_path_src):
                    if not file_name.endswith('.flac'):
                        continue
                    # Load audio file (keep only first channel).
                    file_path_src = os.path.join(book_path_src, file_name)
                    audio, sr = torchaudio.load(file_path_src, channels_first=True)
                    audio = audio[0]
                    # Resample to 16kHz and center around zero.
                    resampler = tt.Resample(orig_freq=sr, new_freq=SR)
                    audio = resampler(audio)
                    audio = center_audio_tensor(audio)
                    # Find beginning and end of utterance.
                    first_index, last_index = find_first_last_index(audio)
                    first_index = max(0, first_index - int(SR*random.uniform(0.2, 0.5)))
                    last_index = min(audio.size(0), last_index + int(SR*random.uniform(0.2, 0.5)))
                    audio = audio[first_index:last_index]
                    # Pad the audio segment to 4s.
                    audio = pad_and_slice_audio_segment(audio, SR, target_len_in_s=4.0, fade_in_len_in_s=random.uniform(0.1, 0.4), fade_out_len_in_s=random.uniform(0.1, 0.4))
                    # Fade and normalize.
                    fader = tt.Fade(fade_in_len=1024, fade_out_len=1024, fade_shape='linear')                
                    audio = fader(audio)
                    audio /= audio.abs().max()
                    # Save segment.
                    file_path_dst = os.path.join(speaker_path_dst, file_name)
                    torchaudio.save(file_path_dst, audio.unsqueeze(0), SR)                 


def process_dmnd(dmnd_root, repeats=1):
    """Split the DEMAND database into training, validation and test subsets.
    Load all WAV audio files in DEMAND, slice them in 4-channel segments of length 4s, and save the resulting audio files."""
    
    # Generate and shuffle the list of all noise environments.
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
    # Parse and process each subset.
    for subset, subset_name  in [(trn_subset, 'trn'), (tst_subset, 'tst'), (val_subset, 'val')]:    
        for environment in subset:
            envt_name = environment[:-4]
            envt_path_src = os.path.join(dmnd_root, '16k', environment, envt_name)
            envt_path_dst = os.path.join(os.path.dirname(dmnd_root), 'sliced_dmnd', subset_name, envt_name)
            os.makedirs(envt_path_dst, exist_ok=True)
            for file_name in os.listdir(envt_path_src):
                # Load audio file (keep only first channel).
                file_path_src = os.path.join(envt_path_src, file_name)
                audio, sr = torchaudio.load(file_path_src, channels_first=True)
                audio = audio[0]
                # Resample to 16kHz amd center around zero.
                resampler = tt.Resample(orig_freq=sr, new_freq=SR)
                audio = resampler(audio)
                audio = center_audio_tensor(audio)
                # Slice the audio segment to a multiple of 16s.
                audio_len_in_s = audio.size(0) // SR
                # Repeat several times make database larger.
                for r in range(repeats):
                    audio_sliced = pad_and_slice_audio_segment(audio, SR, target_len_in_s = audio_len_in_s - audio_len_in_s%16)
                    # Slice audio in 4s segments.
                    segments_4s = segment_audio_array(audio_sliced, segment_length_in_s=4, fade_length_in_s=0, sample_rate=sr)
                    segments_4s = segments_4s.reshape(segments_4s.size(0)//4, 4, -1)
                    segments_4s = list(segments_4s)
                    random.shuffle(segments_4s);
                    for i, segment_i in enumerate(segments_4s):
                        # Fade and normalize.
                        fader = tt.Fade(fade_in_len=512, fade_out_len=512, fade_shape='linear')
                        segment_4c = fader(segment_i)
                        segment_4c /= torch.max(segment_4c)
                        # Save 4-channel segment.
                        file_path_dst = os.path.join(envt_path_dst, f'{file_name[:-4]}_{r:02}{i:02}.flac')
                        torchaudio.save(file_path_dst, segment_4c, sr)                


def process_wham(wham_root):
    """Split the WHAM database into training, validation and test subsets.
    Load all audio files in WHAM, combine them in 4-channel segments of length 4s, and save the resulting audio files."""
    
    # Parse and process each subset.
    for (subset_name, new_subset_name) in [('tt', 'tst'), ('tr', 'trn'), ('cv', 'val')]:
        wham_path_src = os.path.join(wham_root, subset_name)
        wham_list = sorted(os.listdir(wham_path_src))
        wham_path_dst = os.path.join(os.path.dirname(wham_root), 'sliced_wham', new_subset_name)
        os.makedirs(wham_path_dst, exist_ok=True) 
        for i, (file_name_1, file_name_2, file_name_3, file_name_4) in enumerate(zip(wham_list[::4], wham_list[1::4], wham_list[2::4], wham_list[3::4])):
            # Load audio files (keep only first channel).
            file_path_1 = os.path.join(wham_path_src, file_name_1)
            file_path_2 = os.path.join(wham_path_src, file_name_2)
            file_path_3 = os.path.join(wham_path_src, file_name_3)
            file_path_4 = os.path.join(wham_path_src, file_name_4)
            audio_1 = torchaudio.load(file_path_1, channels_first=True)[0][0]
            audio_2 = torchaudio.load(file_path_2, channels_first=True)[0][0]
            audio_3 = torchaudio.load(file_path_3, channels_first=True)[0][0]
            audio_4 = torchaudio.load(file_path_4, channels_first=True)[0][0]
            # Center audio files.
            audio_1 = center_audio_tensor(audio_1)
            audio_2 = center_audio_tensor(audio_2)
            audio_3 = center_audio_tensor(audio_3)
            audio_4 = center_audio_tensor(audio_4)
            # Keep only first 4s of each audio file (pad if necessary).
            audio_1 = pad_and_slice_audio_segment(audio_1, SR, target_len_in_s=4.0, padding_mode='reflect')
            audio_2 = pad_and_slice_audio_segment(audio_2, SR, target_len_in_s=4.0, padding_mode='reflect')
            audio_3 = pad_and_slice_audio_segment(audio_3, SR, target_len_in_s=4.0, padding_mode='reflect')
            audio_4 = pad_and_slice_audio_segment(audio_4, SR, target_len_in_s=4.0, padding_mode='reflect')
            # Create 4-channel segment of length 4s, fade and normalize.
            segment_4s = torch.stack((audio_1, audio_2, audio_3, audio_4), dim=0)
            fader = tt.Fade(fade_in_len=int(SR*0.01), fade_out_len=int(SR*0.01), fade_shape='linear')
            segment_4s = fader(segment_4s)
            segment_4s = segment_4s / torch.max(segment_4s) #, dim=1, keepdim=True)[0]
            # Save segment.
            file_path_dst = os.path.join(wham_path_dst, f"{i:05}.flac")
            torchaudio.save(file_path_dst, segment_4s, SR)


def parse_vctk(vctk_root, subset='trn'):
    """Parse database and return shuffled list of all files in the given subset."""
    vctk_path = os.path.join(vctk_root, subset)
    file_list = []

    for speaker_name in os.listdir(vctk_path):
        speaker_path = os.path.join(vctk_path, speaker_name)
        for file_name in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, file_name)
            file_list.append(file_path)
        
    print(f"Found {len(file_list)} files in '{vctk_path}'.")
    random.shuffle(file_list)
    return file_list


def parse_lisp(lisp_root, subset='trn'):
    """Parse database and return shuffled list of all files in the given subset."""
    lisp_path = os.path.join(lisp_root, subset)
    file_list = []

    for speaker_name in os.listdir(lisp_path):
        speaker_path = os.path.join(lisp_path, speaker_name)
        for file_name in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, file_name)
            file_list.append(file_path)
        
    print(f"Found {len(file_list)} files in '{lisp_path}'.")
    random.shuffle(file_list)
    return file_list


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

    # lisp_root = r"/home/ovistetom/Documents/Databases_Local/LISP/LibriSpeech"
    # rearrange_lisp_subsets(lisp_root)

    vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK/VCTK_092"
    lisp_root = r"/home/ovistetom/Documents/Databases_Local/LISP/LibriSpeech"
    dmnd_root = r"/home/ovistetom/Documents/Databases_Local/DMND/DEMAND"
    process_vctk(vctk_root)
    process_lisp(lisp_root)
    process_dmnd(dmnd_root,repeats=20)

    # subset = 'val'
    # vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK/sliced_vctk"
    # lisp_root = r"/home/ovistetom/Documents/Databases_Local/LISP/sliced_lisp"
    # wham_root = r"/home/ovistetom/Documents/Databases_Local/WHAM/sliced_wham"
    # dmnd_root = r"/home/ovistetom/Documents/Databases_Local/DMND/sliced_dmnd" 
    # parse_vctk(vctk_root, subset=subset)
    # parse_lisp(lisp_root, subset=subset)
    # parse_wham(wham_root, subset=subset)
    # parse_dmnd(dmnd_root, subset=subset)  