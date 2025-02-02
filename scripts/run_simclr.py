from training.train_simclr import train_simclr

def run_simclr(batch_size, unlabeled_data, train_data_contrast, num_workers, **kwargs):
    return train_simclr(batch_size=batch_size, unlabeled_data=unlabeled_data, train_data_contrast=train_data_contrast,
                        num_workers=num_workers, max_epochs=500, **kwargs)