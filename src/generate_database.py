import os
import random
from room_acoustics import create_mixture_audio_sample
from handle_databases import parse_vctk, parse_lisp, parse_wham, parse_dmnd


if __name__ == '__main__':

    subset = 'tst'

    vctk_root = r"/home/ovistetom/Documents/Databases_Local/VCTK/sliced_vctk"
    lisp_root = r"/home/ovistetom/Documents/Databases_Local/LISP/sliced_lisp"
    wham_root = r"/home/ovistetom/Documents/Databases_Local/WHAM/sliced_wham"
    dmnd_root = r"/home/ovistetom/Documents/Databases_Local/DMND/sliced_dmnd"

    out = os.path.join(r"/home/ovistetom/Documents/Databases_Local/MIXTURES")

    vctk_list_mic1, vctk_list_mic2 = parse_vctk(vctk_root, subset=subset)
    lisp_list = parse_lisp(lisp_root, subset=subset)
    wham_list = parse_wham(wham_root, subset=subset)
    dmnd_list = parse_dmnd(dmnd_root, subset=subset)
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

