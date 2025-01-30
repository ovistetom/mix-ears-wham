import os
import random
from room_acoustics import create_mixture_audio_sample
from handle_databases import parse_vctk, parse_lisp, parse_wham, parse_dmnd


if __name__ == '__main__':

    vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK_092/wav48_silence_trimmed"
    lisp_root = r"/home/ovistetom/Documents/Databases_Local/LIBRIMIX/LibriSpeech"
    wham_root = r"/home/ovistetom/Documents/Databases_Local/LIBRIMIX/wham_noise"
    dmnd_root = r"/home/ovistetom/Documents/Databases_Local/DEMAND/16k"

    out = os.path.join(r"/home/ovistetom/Documents/Databases_Local/MIXTURES")

    vctk_list_mic1, vctk_list_mic2 = parse_vctk(vctk_root, subset='tst')
    lisp_list = parse_lisp(lisp_root, subset='test-clean')
    wham_list = parse_wham(wham_root, subset='tt')
    dmnd_list = parse_dmnd(dmnd_root, subset='tst')
    nois_list = wham_list + dmnd_list
    random.shuffle(nois_list);

    for i, (file_path_clean_mic1, file_path_clean_mic2, file_path_distr, file_path_noise) in enumerate(zip(vctk_list_mic1, vctk_list_mic2, lisp_list, nois_list)):

        out_i = os.path.join(out, f"{i:04}")
        os.makedirs(out_i, exist_ok=True)

        create_mixture_audio_sample(file_path_clean_mic1, # MIC1 
                                    file_path_distr, 
                                    file_path_noise, 
                                    room_is_anechoic=False, 
                                    path_to_output_folder=out_i)
        
        create_mixture_audio_sample(file_path_clean_mic2, # MIC2
                                    file_path_distr, 
                                    file_path_noise, 
                                    room_is_anechoic=True, 
                                    path_to_output_folder=out_i)

