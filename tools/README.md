# Preprocessing

    # Download MNIST-M
    http://yaroslav.ganin.net/

    # build classification dataset
    python create_fivesix_bias_dataset.py

    # resize images
    python change_image_size.py

    # convert classification dataset to AB dataset (for image translation)
    python convert_classifier_dataset_to_AB_dataset.py

    # build match.csv for later use
    python construct_random_pair.py