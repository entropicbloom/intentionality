# LightningModule that receives a PyTorch model as input
import torch
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MeanSquaredError


class LightningClassificationModel(pl.LightningModule):
    def __init__(self, model, learning_rate, num_classes):
        super().__init__()
        
        # Store model and hyperparameters
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])
        
        # Optional: get dropout probability if it exists
        self.dropout_proba = getattr(model, "dropout_proba", None)
        
        # Initialize metrics - use different names to avoid conflicts
        metrics = {"train_metrics": None, "valid_metrics": None, "test_metrics": None}
        for split in metrics:
            metrics[split] = torchmetrics.Accuracy(
                task='multiclass', 
                num_classes=num_classes
            )
        self.metrics = torch.nn.ModuleDict(metrics)

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        true_labels = true_labels.squeeze().long()
        logits = self(features)

        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        # Compute accuracy in eval mode
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.metrics["train_metrics"](predicted_labels, true_labels)
        self.log("train_acc", self.metrics["train_metrics"], on_epoch=True, on_step=False)
        self.model.train()
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.metrics["valid_metrics"](predicted_labels, true_labels)
        self.log(
            "valid_acc",
            self.metrics["valid_metrics"],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.metrics["test_metrics"](predicted_labels, true_labels)
        self.log("test_acc", self.metrics["test_metrics"], on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LightningRegressionModel(pl.LightningModule):
    """PyTorch Lightning module for regression tasks."""
    def __init__(self, model, learning_rate, label_dim):
        """
        Args:
            model: The PyTorch model to train.
            learning_rate: The learning rate for the optimizer.
            label_dim: The dimension of the output labels.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.label_dim = label_dim
        self.save_hyperparameters(ignore=["model"])

        # Initialize MeanSquaredError metric for train, validation, and test
        metrics = {"train_metrics": None, "valid_metrics": None, "test_metrics": None}
        for split in metrics:
            metrics[split] = MeanSquaredError()
        self.metrics = torch.nn.ModuleDict(metrics)
        self.metric_log_name = 'mse'

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        predictions = self(features)

        # Ensure labels are float and correct shape for MSE
        true_labels = true_labels.float().squeeze()
        # Handle potential extra dimension in predictions
        if predictions.ndim == true_labels.ndim + 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        # Ensure predictions require grad if needed (usually they do)
        # if not predictions.requires_grad: predictions.requires_grad_(True) # Might not be needed
            
        loss = torch.nn.functional.mse_loss(predictions, true_labels)
        # For MSE, the prediction itself is used for metric calculation
        predicted_values = predictions 

        return loss, true_labels, predicted_values

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_values = self._shared_step(batch)
        self.log("train_loss", loss)
        self.metrics["train_metrics"].update(predicted_values, true_labels)
        self.log(f"train_{self.metric_log_name}", self.metrics["train_metrics"], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_values = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.metrics["valid_metrics"].update(predicted_values, true_labels)
        self.log(
            f"valid_{self.metric_log_name}",
            self.metrics["valid_metrics"],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_values = self._shared_step(batch)
        self.metrics["test_metrics"].update(predicted_values, true_labels)
        self.log(f"test_{self.metric_log_name}", self.metrics["test_metrics"], on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer