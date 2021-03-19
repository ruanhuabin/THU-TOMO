void weight_back_projection(MRC* stack_orig, MRC* stack_recon, std::string temp_save_path, int h_tilt_max, int defocus_step,
                            int h, float pix, float Cs, float volt, float w_cos, float psi_deg, int batch_recon, int batch_write,
                            bool flip_contrast, CTF *ctf_para, float *theta, bool ram);
