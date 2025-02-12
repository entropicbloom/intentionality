

def get_dir_path(model_class_str, dataset_class_str, num_epochs, varying_dim, models_dir):
    untrained_str = '-untrained' if num_epochs == 0 else ''
    varying_dim_str = '-varying-dim' if varying_dim else ''
    path = f'{models_dir}{model_class_str}-{dataset_class_str}{untrained_str}{varying_dim_str}/'
    return path