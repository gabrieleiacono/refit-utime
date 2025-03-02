import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import List, Optional
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score


class ConvBlock(nn.Module):
    """
    Convolutional block with two dilated convolutions and batch normalization
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 dilation: int = 9):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding='same', dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding='same', dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling, concatenation with skip connection,
    and two convolutions with batch normalization
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 kernel_size: int = 5, up_kernel_size: Optional[int] = None):
        super().__init__()
        self.up_kernel_size = up_kernel_size if up_kernel_size else kernel_size

        # Upsampling convolution
        self.up_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=self.up_kernel_size,
            padding='same'
        )
        self.bn_up = nn.BatchNorm1d(out_channels)

        # Convolutions after concatenation
        self.conv1 = nn.Conv1d(
            out_channels + skip_channels, out_channels, kernel_size=kernel_size,
            padding='same'
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, skip):
        # Use nearest-neighbor upsampling followed by convolution
        # instead of transposed convolution to avoid checkerboard artifacts
        x = F.interpolate(x, size=skip.shape[2], mode='nearest')
        x = F.relu(self.bn_up(self.up_conv(x)))

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Apply convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class UTime(nn.Module):
    """
    U-Time: A Fully Convolutional Network for Time Series Segmentation
    """

    def __init__(
            self,
            in_channels: int = 1,
            n_classes: int = 2,
            init_filters: int = 16,
            depth: int = 4,
            kernel_size: int = 5,
            dilation: int = 9,
            dropout: float = 0.0,
            pool_sizes: List[int] = None,
            up_kernel_sizes: List[int] = None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.init_filters = init_filters
        self.depth = depth

        # Default pool sizes as in the paper
        if pool_sizes is None:
            self.pool_sizes = [10, 8, 6, 4]
        else:
            self.pool_sizes = pool_sizes

        # Default upsampling kernel sizes as in the paper
        if up_kernel_sizes is None:
            self.up_kernel_sizes = [4, 6, 8, 10]
        else:
            self.up_kernel_sizes = up_kernel_sizes

        # Ensure we have the right number of pool sizes and up kernel sizes
        assert len(self.pool_sizes) >= depth, "Not enough pool sizes"
        assert len(self.up_kernel_sizes) >= depth, "Not enough upsampling kernel sizes"

        # Encoder pathway
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        # Input block
        self.input_block = ConvBlock(in_channels, init_filters, kernel_size, dilation)

        # Encoder blocks
        filters = init_filters
        for i in range(depth):
            # Double the number of filters at each layer
            next_filters = filters * 2

            # Create encoder block
            self.encoder_blocks.append(
                ConvBlock(filters, next_filters, kernel_size, dilation)
            )

            # Create max pooling layer
            self.pool_layers.append(
                nn.MaxPool1d(self.pool_sizes[i])
            )

            filters = next_filters

        # Bottom (lowest resolution) convolutions
        self.bottom_conv1 = nn.Conv1d(filters, filters, kernel_size, padding='same',
                                      dilation=dilation)
        self.bottom_bn1 = nn.BatchNorm1d(filters)
        self.bottom_conv2 = nn.Conv1d(filters, filters, kernel_size, padding='same',
                                      dilation=dilation)
        self.bottom_bn2 = nn.BatchNorm1d(filters)

        # Decoder pathway
        self.decoder_blocks = nn.ModuleList()

        # Decoder blocks
        for i in range(depth):
            # Decoder works in reverse order
            idx = depth - i - 1
            skip_filters = init_filters * (2 ** idx)
            in_filters = filters
            out_filters = filters // 2

            self.decoder_blocks.append(
                DecoderBlock(
                    in_filters, skip_filters, out_filters,
                    kernel_size, self.up_kernel_sizes[i]
                )
            )

            filters = out_filters

        # Output layer
        self.output_layer = nn.Conv1d(filters, n_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)

        # Initial convolution
        x = self.input_block(x)

        # Store skip connections
        skips = [x]

        # Encoder path
        for i, (encoder, pool) in enumerate(zip(self.encoder_blocks, self.pool_layers)):
            # Apply encoder block
            x = encoder(x)
            # Store the result for skip connection
            skips.append(x)
            # Apply max pooling
            x = pool(x)

        # Bottom convolutions
        x = F.relu(self.bottom_bn1(self.bottom_conv1(x)))
        x = F.relu(self.bottom_bn2(self.bottom_conv2(x)))

        # Decoder path
        for i, decoder in enumerate(self.decoder_blocks):
            # Get the skip connection from the encoder
            skip_idx = len(skips) - i - 2
            skip = skips[skip_idx]
            # Apply decoder block
            x = decoder(x, skip)

        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)

        return x


class SegmentClassifier(nn.Module):
    """
    Segment classifier that aggregates dense time point classifications
    to produce segment-level classifications
    """

    def __init__(self, n_classes: int = 2, segment_size: int = 3000):
        super().__init__()
        self.n_classes = n_classes
        self.segment_size = segment_size
        self.segment_conv = nn.Conv1d(n_classes, n_classes, kernel_size=1)

    def forward(self, x, segment_size=None):
        # If segment_size is not provided, use the default
        if segment_size is None:
            segment_size = self.segment_size

        # Apply average pooling over segments
        # Shape before: (batch, n_classes, sequence_length)
        x = F.avg_pool1d(x, kernel_size=segment_size, stride=segment_size)
        # Shape after: (batch, n_classes, n_segments)

        # Apply final 1x1 convolution to re-weight class confidence scores
        x = self.segment_conv(x)
        # Apply softmax along class dimension
        x = F.softmax(x, dim=1)

        return x


class UTimeModel(pl.LightningModule):
    """
    PyTorch Lightning module for U-Time
    """

    def __init__(
            self,
            in_channels: int = 1,
            n_classes: int = 2,
            segment_size: int = 3000,
            learning_rate: float = 5e-6,
            log_confusion_matrix: bool = True,
            log_predictions: bool = True,
            log_gradients: bool = True,
            log_model_graph: bool = True,
            **utime_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create the U-Time model
        self.utime = UTime(in_channels=in_channels, n_classes=n_classes, **utime_kwargs)

        # Create the segment classifier
        self.segment_classifier = SegmentClassifier(n_classes, segment_size)

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.n_classes = n_classes

        # Logging settings
        self.log_confusion_matrix = log_confusion_matrix
        self.log_predictions = log_predictions
        self.log_gradients = log_gradients
        self.log_model_graph = log_model_graph

        # For tracking metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def get_model_graph(self, save_path='utime_model_graph.png',
                        sample_input_size=(1, 1, 3000 * 35), dpi=300):
        """
        Generate and save a visualization of the model's computational graph

        Parameters:
        -----------
        save_path : str
            Path where the model graph image will be saved
        sample_input_size : tuple
            Size of the sample input tensor (batch_size, channels, sequence_length)
        dpi : int
            Resolution of the output image

        Returns:
        --------
        Path to the saved image

        Note: Requires torchviz and graphviz packages
        """
        import torch
        from torchviz import make_dot

        # Create a sample input tensor
        sample_input = torch.zeros(sample_input_size)

        # Get the output of the model
        output = self(sample_input)

        # Create the graph
        dot = make_dot(output, params=dict(self.named_parameters()), show_attrs=True,
                       show_saved=True)

        # Set graph attributes
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr(dpi=str(dpi))

        # Save the graph
        dot.render(save_path.replace('.png', ''), format='png', cleanup=True)

        print(f"Model graph saved to {save_path}")
        return save_path

    def forward(self, x):
        # Get dense predictions from U-Time
        x = self.utime(x)
        # Apply segment classifier
        x = self.segment_classifier(x)
        return x

    def configure_optimizers(self):
        # Use Adam optimizer as in the paper
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def generalized_dice_loss(self, y_pred, y_true):
        """
        Generalized Dice Loss with uniform class weights as used in the paper
        """
        # Apply softmax to get class probabilities
        y_pred = F.softmax(y_pred, dim=1)

        # Flatten predictions and targets
        y_pred = y_pred.reshape(-1, self.n_classes)  # Changed from view() to reshape()
        y_true = y_true.reshape(-1, self.n_classes)  # Changed from view() to reshape()

        # Calculate intersection and union
        intersection = torch.sum(y_pred * y_true, dim=0)
        union = torch.sum(y_pred + y_true, dim=0)

        # Calculate Dice score for each class
        dice = (2.0 * intersection) / (union + 1e-8)

        # Average Dice score over all classes
        mean_dice = torch.mean(dice)

        # Return Dice loss
        return 1.0 - mean_dice

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        # One-hot encode the targets
        y_onehot = F.one_hot(y, num_classes=self.n_classes).permute(0, 2, 1).float()

        # Calculate loss
        loss = self.generalized_dice_loss(y_pred, y_onehot)

        # Get predicted class
        y_pred_class = torch.argmax(y_pred, dim=1)

        # Calculate accuracy
        accuracy = (y_pred_class == y).float().mean()

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # Store outputs for epoch end processing
        self.training_step_outputs.append({
            'loss': loss,
            'y_pred': y_pred_class.detach(),
            'y': y.detach()
        })

        # Log parameter histograms (once per epoch to avoid slowdown)
        if self.log_gradients and batch_idx == 0:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"gradients/{name}",
                        param.grad.data.cpu().numpy(),
                        self.current_epoch
                    )
                    self.logger.experiment.add_histogram(
                        f"weights/{name}",
                        param.data.cpu().numpy(),
                        self.current_epoch
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        # One-hot encode the targets
        y_onehot = F.one_hot(y, num_classes=self.n_classes).permute(0, 2, 1).float()

        # Calculate loss
        loss = self.generalized_dice_loss(y_pred, y_onehot)

        # Get predicted class
        y_pred_class = torch.argmax(y_pred, dim=1)

        # Calculate accuracy
        accuracy = (y_pred_class == y).float().mean()

        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

        # Store outputs for epoch end processing
        self.validation_step_outputs.append({
            'loss': loss,
            'y_pred': y_pred_class.detach(),
            'y': y.detach(),
            'x': x.detach() if self.log_predictions and batch_idx == 0 else None
        })

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        # One-hot encode the targets
        y_onehot = F.one_hot(y, num_classes=self.n_classes).permute(0, 2, 1).float()

        # Calculate loss
        loss = self.generalized_dice_loss(y_pred, y_onehot)

        # Get predicted class
        y_pred_class = torch.argmax(y_pred, dim=1)

        # Calculate accuracy
        accuracy = (y_pred_class == y).float().mean()

        # Log test metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', accuracy, on_epoch=True)

        # Store outputs for epoch end processing
        self.test_step_outputs.append({
            'loss': loss,
            'y_pred': y_pred_class.detach(),
            'y': y.detach()
        })

        return loss

    def on_train_epoch_end(self):
        # Only run if we have a logger
        if not self.logger:
            self.training_step_outputs.clear()
            return

        # Process outputs for additional logging
        y_pred = torch.cat([x['y_pred'] for x in self.training_step_outputs]).cpu().numpy()
        y_true = torch.cat([x['y'] for x in self.training_step_outputs]).cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
        mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())

        # Log metrics
        self.log('train_f1', f1, on_epoch=True)
        self.log('train_mcc', mcc, on_epoch=True)

        # Log confusion matrix
        if self.log_confusion_matrix:
            self._log_confusion_matrix(y_true, y_pred, 'train')

        # Clear outputs to free memory
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        # Only run if we have a logger
        if not self.logger:
            self.validation_step_outputs.clear()
            return

        # Process outputs for additional logging
        y_pred = torch.cat(
            [x['y_pred'] for x in self.validation_step_outputs]).cpu().numpy()
        y_true = torch.cat([x['y'] for x in self.validation_step_outputs]).cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
        mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())

        # Log metrics
        self.log('val_f1', f1, on_epoch=True)
        self.log('val_mcc', mcc, on_epoch=True)

        # Log confusion matrix
        if self.log_confusion_matrix:
            self._log_confusion_matrix(y_true, y_pred, 'val')

        # Log sample predictions
        if self.log_predictions:
            # Find a batch with input data
            x_batch = None
            for output in self.validation_step_outputs:
                if output['x'] is not None:
                    x_batch = output['x']
                    break

            if x_batch is not None:
                self._log_predictions(x_batch, 'val')

        # Clear outputs to free memory
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        # Only run if we have a logger
        if not self.logger:
            self.test_step_outputs.clear()
            return

        # Process outputs for additional logging
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs]).cpu().numpy()
        y_true = torch.cat([x['y'] for x in self.test_step_outputs]).cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
        mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())

        # Log metrics
        self.log('test_accuracy', accuracy, on_epoch=True)
        self.log('test_f1', f1, on_epoch=True)
        self.log('test_mcc', mcc, on_epoch=True)

        # Print metrics to console for easier viewing during testing
        print(
            f"Test Metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}")

        # Log confusion matrix
        if self.log_confusion_matrix:
            self._log_confusion_matrix(y_true, y_pred, 'test')

        # If you want to log detailed per-class metrics
        if self.n_classes > 2:
            # For multi-class, log per-class F1
            class_f1 = f1_score(y_true.flatten(), y_pred.flatten(), average=None)
            for i, score in enumerate(class_f1):
                self.log(f'test_f1_class_{i}', score, on_epoch=True)
        else:
            # For binary classification, log additional metrics
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(y_true.flatten(), y_pred.flatten())
            recall = recall_score(y_true.flatten(), y_pred.flatten())
            self.log('test_precision', precision, on_epoch=True)
            self.log('test_recall', recall, on_epoch=True)

        # Clear outputs to free memory
        self.test_step_outputs.clear()

    def _log_confusion_matrix(self, y_true, y_pred, stage):
        """
        Log confusion matrix to TensorBoard
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # Create confusion matrix
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create figure and plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Normalized Confusion Matrix - {stage}')

        # Log figure to TensorBoard
        self.logger.experiment.add_figure(
            f'{stage}_confusion_matrix',
            fig,
            global_step=self.current_epoch
        )
        plt.close(fig)

    def _log_predictions(self, x_batch, stage):
        """
        Log sample predictions to TensorBoard
        """
        # Get predictions
        with torch.no_grad():
            y_pred = self(x_batch)
            y_pred_class = torch.argmax(y_pred, dim=1)

        # Take only the first few samples
        max_samples = min(4, x_batch.size(0))
        x = x_batch[:max_samples].cpu()
        y_pred = y_pred_class[:max_samples].cpu()

        # Create figure for each sample
        for i in range(max_samples):
            fig, ax = plt.subplots(figsize=(15, 5))

            # Plot the time series
            ax.plot(x[i, 0, :].numpy())

            # Create a colormap for segments
            segment_size = self.hparams.segment_size
            n_segments = x.size(2) // segment_size

            # Plot segment predictions as background colors
            for seg in range(n_segments):
                start = seg * segment_size
                end = (seg + 1) * segment_size
                pred_class = y_pred[i, seg].item()
                color = plt.cm.tab10(pred_class)
                ax.axvspan(start, end, alpha=0.2, color=color)

            ax.set_title(f'Sample {i + 1} with Segment Predictions')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')

            # Add a legend for classes
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=plt.cm.tab10(j), alpha=0.2, label=f'Class {j}')
                for j in range(self.n_classes)
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            # Log figure to TensorBoard
            self.logger.experiment.add_figure(
                f'{stage}_prediction_sample_{i}',
                fig,
                global_step=self.current_epoch
            )
            plt.close(fig)

    def on_fit_start(self):
        # Log model graph
        if self.log_model_graph and hasattr(self.logger, 'experiment'):
            # Create a small sample input for the model graph
            sample_size = 3000  # Use a smaller size for the graph
            sample_input = torch.zeros((1, self.hparams.in_channels, sample_size))
            self.logger.experiment.add_graph(self, sample_input)


# Example dataset class - modify with your actual time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, segment_size=3000):
        self.data = data
        self.targets = targets
        self.segment_size = segment_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        # Ensure x is the right shape (channels, sequence_length)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        return x, y


# Example usage:
def train_utime_model(
        train_data,
        train_targets,
        val_data=None,
        val_targets=None,
        test_data=None,
        test_targets=None,
        segment_size=3000,
        batch_size=12,
        max_epochs=100,
        gpus=1,
        log_dir="lightning_logs",
        experiment_name="utime_run",
        **model_kwargs
):
    """
    Train a U-Time model with the given data

    Parameters:
    -----------
    train_data : np.ndarray
        Training data of shape (n_samples, seq_length) or (n_samples, n_channels, seq_length)
    train_targets : np.ndarray
        Training targets of shape (n_samples, n_segments)
    val_data : np.ndarray, optional
        Validation data
    val_targets : np.ndarray, optional
        Validation targets
    test_data : np.ndarray, optional
        Test data
    test_targets : np.ndarray, optional
        Test targets
    segment_size : int
        Size of segments for classification
    batch_size : int
        Batch size for training
    max_epochs : int
        Maximum number of training epochs
    gpus : int
        Number of GPUs to use
    log_dir : str
        Directory for TensorBoard logs
    experiment_name : str
        Name of the experiment for TensorBoard
    model_kwargs : dict
        Additional arguments to pass to the UTimeModel constructor

    Returns:
    --------
    model : UTimeModel
        Trained U-Time model
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, train_targets, segment_size)

    if val_data is not None and val_targets is not None:
        val_dataset = TimeSeriesDataset(val_data, val_targets, segment_size)
    else:
        # If no validation data is provided, split the training data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    if test_data is not None and test_targets is not None:
        test_dataset = TimeSeriesDataset(test_data, test_targets, segment_size)
    else:
        test_dataset = None

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    else:
        test_loader = None

    # Determine input channels from data
    sample_x, _ = train_dataset[0]
    in_channels = sample_x.shape[0]

    # Set up TensorBoard logger
    logger = TensorBoardLogger(log_dir, name=experiment_name)

    # Create model
    model = UTimeModel(
        in_channels=in_channels,
        segment_size=segment_size,
        **model_kwargs
    )

    # Create callbacks
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=150,  # As in the paper
            mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.4f}',
            save_top_k=3,
            mode='min'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    # Create trainer with TensorBoard logger
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        # accelerator='gpu' if gpus > 0 else 'cpu',
        # devices=gpus if gpus > 0 else None,
        log_every_n_steps=10
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Test model if test data is available
    if test_loader:
        trainer.test(model, test_loader)

    return model


# Example usage
if __name__ == "__main__":
    # Generate some dummy data
    np.random.seed(42)
    n_samples = 100
    seq_length = 3000 * 35  # As in the paper: 35 segments of 3000 points each
    n_channels = 1
    n_segments = seq_length // 3000

    # Create dummy data
    data = np.random.randn(n_samples, n_channels, seq_length)
    targets = np.random.randint(0, 2, (n_samples, n_segments))

    # Convert to torch tensors
    data = torch.tensor(data, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    # Split into train, val, test
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size

    train_data = data[:train_size]
    train_targets = targets[:train_size]

    val_data = data[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]

    test_data = data[train_size + val_size:]
    test_targets = targets[train_size + val_size:]

    # Train model
    model = train_utime_model(
        train_data=train_data,
        train_targets=train_targets,
        val_data=val_data,
        val_targets=val_targets,
        test_data=test_data,
        test_targets=test_targets,
        segment_size=3000,
        batch_size=12,
        max_epochs=100,
        gpus=0,
        n_classes=2,
        init_filters=16,
        depth=4,
        learning_rate=5e-6,
        log_dir="lightning_logs",
        experiment_name="utime_sleep_staging"
    )
