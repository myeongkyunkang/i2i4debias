# i2i4debias
Neural Networks. "Content preserving image translation with texture co-occurrence and spatial self-similarity for texture debiasing and domain adaptation"

# Train Image Translation Model

    # train fivesix AtoB
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --dataroot ./dataset/fivesix_bias_dataset_32_AB \
        --name AtoB \
        --direction AtoB \
        --checkpoints_dir result_fivesix \
        --dataset_mode unaligned \
        --netE_num_downsampling_sp 1 \
        --netE_num_downsampling_gl 3 \
        --load_size 32 \
        --crop_size 32 \
        --batch_size 8 \
        --total_nimgs 1200000 \
        --no_flip \
        --preprocess custom_noaffine

    # train fivesix BtoA
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --dataroot ./dataset/fivesix_bias_dataset_32_AB \
        --name BtoA \
        --direction BtoA \
        --checkpoints_dir result_fivesix \
        --dataset_mode unaligned \
        --netE_num_downsampling_sp 1 \
        --netE_num_downsampling_gl 3 \
        --load_size 32 \
        --crop_size 32 \
        --batch_size 8 \
        --total_nimgs 1200000 \
        --no_flip \
        --preprocess custom_noaffine

# Translate Images

    # translate fivesix AtoB
    CUDA_VISIBLE_DEVICES=0 python test_with_match.py \
        --dataroot ./dataset/fivesix_bias_dataset_32_AB \
        --name AtoB \
        --direction AtoB \
        --netE_num_downsampling_sp 1 \
        --netE_num_downsampling_gl 3 \
        --load_size 32 \
        --crop_size 32 \
        --match tools/fivesix/match.csv \
        --resume_iter latest \
        --checkpoints_dir result_fivesix \
        --result_dir result_fivesix

    # translate fivesix BtoA
    CUDA_VISIBLE_DEVICES=0 python test_with_match.py \
        --dataroot ./dataset/fivesix_bias_dataset_32_AB \
        --name BtoA \
        --direction BtoA \
        --netE_num_downsampling_sp 1 \
        --netE_num_downsampling_gl 3 \
        --load_size 32 \
        --crop_size 32 \
        --match tools/fivesix/match.csv \
        --resume_iter latest \
        --checkpoints_dir result_fivesix \
        --result_dir result_fivesix

# Preparation

    Check tools

# Citation
If you find this repository useful in your research, please cite:
```
@article{kang2023content,
  title={Content preserving image translation with texture co-occurrence and spatial self-similarity for texture debiasing and domain adaptation},
  author={Kang, Myeongkyun and Won, Dongkyu and Luna, Miguel and Chikontwe, Philip and Hong, Kyung Soo and Ahn, June Hong and Park, Sang Hyun},
  journal={Neural Networks},
  volume={166},
  pages={722--737},
  year={2023},
  publisher={Elsevier}
}
```

Thanks to works below for their implementations which were useful for this work.
[Swapping Autoencoder](https://github.com/taesungp/swapping-autoencoder-pytorch)
