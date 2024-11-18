class Config:
    # General settings
    seed = 42
    num_workers = 4

    # SimCLR settings
    simclr_hidden_dim = 128
    simclr_lr = 1e-3
    simclr_weight_decay = 1e-4
    simclr_temperature = 0.07
    simclr_max_epochs = 100

    # MLP classifier settings
    classifier_hidden_dim = 512
    classifier_lr = 3e-4
    classifier_weight_decay = 1e-5
    classifier_max_epochs = 100
    classifier_batch_size = 256

    # Dataset paths
    data_path = "data/"
    saved_models_path = "saved_models/"
