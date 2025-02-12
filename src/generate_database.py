import os
from room_acoustics import create_mixture_audio_sample
from preprocess_databases import parse_vctk, parse_lisp, parse_dmnd
from tqdm import tqdm

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

        for i, (file_path_truth, file_path_distr, file_path_noise) in enumerate(tqdm(zip(vctk_list, lisp_list, dmnd_list), total=len(vctk_list), desc=f"Processing subset '{subset}'")):

            out_i = os.path.join(out_root, subset, f"{i:05}")
            # Define output path.
            os.makedirs(os.path.join(out_i, 'echoFalse'), exist_ok=True)
            os.makedirs(os.path.join(out_i, 'echoTrue'), exist_ok=True)
            # Create mixture.
            create_mixture_audio_sample(file_path_truth,
                                        file_path_distr, 
                                        file_path_noise, 
                                        room_is_anechoic=False, 
                                        path_to_output_folder=os.path.join(out_i, 'echoFalse'),
                                        target_length_in_s=4.0,
                                        )
            create_mixture_audio_sample(file_path_truth,
                                        file_path_distr, 
                                        file_path_noise, 
                                        room_is_anechoic=True, 
                                        path_to_output_folder=os.path.join(out_i, 'echoTrue'),
                                        target_length_in_s=4.0,
                                        )

