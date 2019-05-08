data_params = {
    "width"                 : 640,
    "height"                : 480,
    "labelled_train_db"     : 'data/labels.train.db',
    "labelled_train_dir"    : 'data/labelled.train/',
    "labelled_validate_db"  : 'data/labels.validate.db',
    "labelled_validate_dir" : 'data/labelled.validate/',
    "segmented_dir"         : 'data/segmented/'
}

model_params = {
    "num_classes" : 2,
    "batch_size"  : 300,
    "epochs"      : 20
}

hyper_params = {
    "max_training_images_count"    : 100,     # if value is -1 then take all training images
    "max_testing_images_count"     : -1,      # if value is -1 then take all validation images
    "l2_regularisation"            : 0.0005,
    "dropout"                      : 0.50,
    "learning_rate"                : 0.0001,
    "percentage_train"             : 80,
    "windows_per_image_on_average" : 80,
    "window_size"                  : 32,
    "bee_radius"                   : 16,
    "min_bee_prob"                 : 0.99,
    "max_bee_prob"                 : 1.00,
    "min_bee_prct_window"          : 50.0,
    "bee_window_percentage"        : 20,
    "filters_count"                : 32,
    "kernel_size"                  : 3,
    "padding_to_remove"            : 0,
    "sliding_window_step"          : 2
}
