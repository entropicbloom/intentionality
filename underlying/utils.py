def get_dir_path(model_class_str, dataset_class_str, num_epochs, hidden_dim, varying_dim, models_dir):
    untrained_str = '-untrained' if num_epochs == 0 else ''
    hidden_dim_str = f"-hidden_dim_{str(hidden_dim).replace(' ', '')}" if not varying_dim else '' # Only include if not varying
    varying_dim_str = '-varying-dim' if varying_dim else ''
    path = f'{models_dir}{model_class_str}-{dataset_class_str}{untrained_str}{hidden_dim_str}{varying_dim_str}/'
    return path