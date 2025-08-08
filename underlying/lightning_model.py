# LightningModule that receives a PyTorch model as input
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate, num_classes):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        # Set up attributes for computing the accuracy
        if not self.model.generative:
            self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
            self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
            self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        else:
            self.train_acc = torchmetrics.MeanSquaredError()
            self.valid_acc = torchmetrics.MeanSquaredError()
            self.test_acc = torchmetrics.MeanSquaredError()

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):

        if not self.model.generative:
            features, true_labels = batch
            logits = self(features)

            loss = torch.nn.functional.cross_entropy(logits, true_labels)
            predicted_labels = torch.argmax(logits, dim=1)

            return loss, true_labels, predicted_labels
        else:
            features, true_labels = batch 
            true_labels = F.one_hot(true_labels, num_classes=self.model.input_dim).type(torch.float32)
            pred = self(true_labels)
            loss_fn = torch.nn.MSELoss()

            loss = loss_fn(pred, features.view(-1, self.model.output_dim))

            return loss, features, pred

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        # Do another forward pass in .eval() mode to compute accuracy
        # while accountingfor Dropout, BatchNorm etc. behavior
        # during evaluation (inference)
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        if not self.model.generative:
            self.train_acc(predicted_labels, true_labels)
        else:
            self.train_acc(predicted_labels, true_labels.view(-1, self.model.output_dim))
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()

        return loss  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        if not self.model.generative:
            self.valid_acc(predicted_labels, true_labels)
        else:
            self.valid_acc(predicted_labels, true_labels.view(-1, self.model.output_dim))
        self.log(
            "valid_acc",
            self.valid_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
