import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from decoder.underlying_datasets import LastLayerDataModule, MixedHiddenDimsDataModule
from decoder.lightning_model import LightningClassificationModel
from underlying.utils import get_dir_path
from decoder.models import decoder_dict
from decoder.config import get_underlying_path

def setup_and_train(seed, num_neurons, project_name, config):
    """Sets up and trains a decoder model for class ID prediction."""
    torch.manual_seed(seed)

    # Use get_dir_path to create the dataset path
    dataset_path = get_underlying_path(get_dir_path(
        model_class_str=config['model_class_str'],
        dataset_class_str=config['dataset_class_str'],
        num_epochs=0 if config.get('untrained', False) else 2,
        hidden_dim=config.get('hidden_dim', [50, 50]),
        varying_dim=config.get('varying_dim', False),
        models_dir=config.get('models_dir', 'saved_models/')
    ))

    # Get the configuration string for wandb naming (reverted)
    underlying_config_str = dataset_path.split('/')[-2]  # Extract the directory name

    # Initialize wandb with the provided project name
    decoder_class = config.get('decoder_class', 'TransformerDecoder')
    wandb.init(
        project=project_name,
        config=config,
        name=f"{underlying_config_str}-{decoder_class}-n{num_neurons}-s{seed}",
        group=f"{underlying_config_str}-{decoder_class}-n{num_neurons}"
    )

    # Create a list of neuron indices to use
    use_neurons = list(range(num_neurons))

    # Initialize model using decoder_dict from models.py
    pytorch_model = decoder_dict[decoder_class](
        dim_input=num_neurons,  # Update the input dimension to match number of neurons
        num_outputs=1,
        dim_output=10,
        num_inds=16,
        dim_hidden=64,
        num_heads=4,
        ln=False
    )

    # Setup training
    lightning_model = LightningClassificationModel(pytorch_model, learning_rate=0.001, num_classes=10)

    # Calculate layer index based on number of hidden layers
    # hidden_dim=[50, 50] -> 2 hidden layers -> output at index 2
    # hidden_dim=[100] -> 1 hidden layer -> output at index 1
    hidden_dim = config.get('hidden_dim', [50, 50])
    layer_idx = len(hidden_dim)

    data_module = LastLayerDataModule(
        dataset_path,
        layer_idx=layer_idx,
        input_dim=50,
        batch_size=64,
        num_workers=0,
        transpose_weights=False,
        preprocessing=config.get('preprocessing', 'multiply_transpose'),
        use_neurons=use_neurons,  # Pass the list of neurons to use
        use_target_similarity_only=config.get('use_target_similarity_only', False),
    )

    # Training configuration
    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")]
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        deterministic=False,
        log_every_n_steps=10,
        logger=WandbLogger()
    )

    # Train model
    trainer.fit(model=lightning_model, datamodule=data_module)
    wandb.finish()

def setup_and_train_mixed_hidden_dims(seed, train_config, valid_config, train_samples, valid_samples, project_name):
    """Sets up and trains a decoder model with different hidden dimensions for train and validation sets."""
    torch.manual_seed(seed)

    # Get dataset paths for both configurations
    train_dataset_path = get_underlying_path(get_dir_path(
        model_class_str=train_config['model_class_str'],
        dataset_class_str=train_config['dataset_class_str'],
        num_epochs=0 if train_config.get('untrained', False) else 2,
        hidden_dim=train_config.get('hidden_dim', [50, 50]),
        varying_dim=train_config.get('varying_dim', False),
        models_dir=train_config.get('models_dir', 'saved_models/')
    ))

    valid_dataset_path = get_underlying_path(get_dir_path(
        model_class_str=valid_config['model_class_str'],
        dataset_class_str=valid_config['dataset_class_str'],
        num_epochs=0 if valid_config.get('untrained', False) else 2,
        hidden_dim=valid_config.get('hidden_dim', [50, 50]),
        varying_dim=valid_config.get('varying_dim', False),
        models_dir=valid_config.get('models_dir', 'saved_models/')
    ))

    # Get configuration strings for wandb naming
    train_config_str = train_dataset_path.split('/')[-2]
    valid_config_str = valid_dataset_path.split('/')[-2]

    # Initialize wandb
    decoder_class = train_config.get('decoder_class', 'TransformerDecoder')
    num_neurons = train_config.get('num_neurons', 10)

    wandb.init(
        project=project_name,
        config={
            "train_config": train_config,
            "valid_config": valid_config,
            "train_samples": train_samples,
            "valid_samples": valid_samples,
            "decoder_class": decoder_class,
            "num_neurons": num_neurons
        },
        name=f"mixed-{train_config_str}-{valid_config_str}-{decoder_class}-n{num_neurons}-s{seed}",
        group=f"mixed-{train_config_str}-{valid_config_str}-{decoder_class}-n{num_neurons}"
    )

    # Create neuron indices
    use_neurons = list(range(num_neurons))

    # Initialize model
    pytorch_model = decoder_dict[decoder_class](
        dim_input=num_neurons,
        num_outputs=1,
        dim_output=10,
        num_inds=16,
        dim_hidden=64,
        num_heads=4,
        ln=False
    )

    # Setup training
    lightning_model = LightningClassificationModel(pytorch_model, learning_rate=0.001, num_classes=10)
    
    # Determine correct layer indices based on hidden dimensions
    # hidden_dim=[100] has final layer at index 1
    # hidden_dim=[50, 50] has final layer at index 2
    train_hidden_dim = train_config.get('hidden_dim', [50, 50])
    valid_hidden_dim = valid_config.get('hidden_dim', [50, 50])
    train_layer_idx = len(train_hidden_dim)  # Number of hidden layers gives final layer index
    valid_layer_idx = len(valid_hidden_dim)

    # Create custom data module with mixed datasets
    data_module = MixedHiddenDimsDataModule(
        train_dataset_path=train_dataset_path,
        valid_dataset_path=valid_dataset_path,
        train_layer_idx=train_layer_idx,
        valid_layer_idx=valid_layer_idx,
        input_dim=50,
        batch_size=64,
        num_workers=0,
        transpose_weights=False,
        preprocessing=train_config.get('preprocessing', 'multiply_transpose'),
        use_neurons=use_neurons,
        use_target_similarity_only=train_config.get('use_target_similarity_only', False),
        train_samples=train_samples,
        valid_samples=valid_samples
    )

    # Training configuration
    callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")]
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        deterministic=False,
        log_every_n_steps=10,
        logger=WandbLogger()
    )

    # Train model
    trainer.fit(model=lightning_model, datamodule=data_module)
    wandb.finish() 