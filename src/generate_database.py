import os
from room_acoustics import create_mixture_audio_sample



if __name__ == '__main__':

    # IDEA: Use VCTK mic1 with echo, use VTCK mic2 without echo... no strong reason, but it allows to use the full VTCK DB, adding a tiny bit more diversity.
    pass

    # create_mixture_audio_sample(file_path_clean, # MIC1 
    #                             file_path_distr, 
    #                             file_path_noise, 
    #                             room_is_anechoic=False, 
    #                             path_to_output_folder=out) 
    
    # create_mixture_audio_sample(file_path_clean, # MIC2
    #                             file_path_distr, 
    #                             file_path_noise, 
    #                             room_is_anechoic=True, 
    #                             path_to_output_folder=out)     