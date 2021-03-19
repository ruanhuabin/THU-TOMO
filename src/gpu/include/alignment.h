void coarse_alignment(int ref_n, int patch_size_coarse, int patch_trans_coarse, int patch_Nx_coarse, int patch_Ny_coarse, int* patch_dx_sum_all, int* patch_dy_sum_all, 
                      MRC* stack_orig, MRC* stack_coarse, std::string patch_save_path);
void patch_tracking(int ref_n, int patch_size, int patch_trans, int patch_Nx, int patch_Ny, MRC* stack_orig, std::string patch_save_path);