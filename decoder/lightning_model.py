# LightningModule that receives a PyTorch model as input
import torch
import pytorch_lightning as pl
import torchmetrics


class LightningModel(pl.LightningModule):
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