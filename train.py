import torch
import numpy as np
import pandas as pd
from datetime import datetime
import pytorch_lightning as L
from torch.utils.data import DataLoader

from utime import UTimeModel

# 1. Load the washing machine data
pdf = pd.read_parquet("washing_machine_data.parquet")
pdf['datetime'] = pd.to_datetime(pdf['Datetime'])

# 2. Split data into train (before 2015-01-01) and test (after)
split_date = datetime(2015, 1, 1)
train_df = pdf[pdf['datetime'] < split_date]
test_df = pdf[pdf['datetime'] >= split_date]

print(f"Training data: {len(train_df)} rows")
print(f"Testing data: {len(test_df)} rows")
print(f"Washing machine active in training: {train_df['washing_machine'].sum()} rows")
print(f"Washing machine active in testing: {test_df['washing_machine'].sum()} rows")

# 3. Process data for U-Time model - we need to segment the time series
segment_size = 100  # Size of each segment for classification


# Custom dataset class to prepare data for U-Time
class WashingMachineDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, segment_size, n_segments_per_sample=10):
        self.df = dataframe
        self.segment_size = segment_size
        self.n_segments_per_sample = n_segments_per_sample
        self.sample_length = segment_size * n_segments_per_sample

        self.samples = []  # Will store fixed-size samples
        self.targets = []  # Will store segment-level targets for each sample

        # Process each house separately
        for house in self.df['house_number'].unique():
            house_df = self.df[self.df['house_number'] == house].sort_values('datetime')

            # Skip houses with too little data
            if len(house_df) < self.sample_length:
                continue

            # Extract features and targets
            features = house_df['Aggregate'].values.astype(np.float32)
            washing_machine = house_df['washing_machine'].astype(int).values

            # Normalize features for better training
            features = (features - features.mean()) / (features.std() + 1e-8)

            # Calculate how many complete samples we can create
            n_samples = len(features) // self.sample_length

            # Process each sample
            for i in range(n_samples):
                start_idx = i * self.sample_length
                end_idx = (i + 1) * self.sample_length

                sample_features = features[start_idx:end_idx]
                sample_targets_raw = washing_machine[start_idx:end_idx]

                # Create segment-level targets
                sample_targets = []
                for j in range(self.n_segments_per_sample):
                    segment_start = j * segment_size
                    segment_end = (j + 1) * segment_size
                    segment_data = sample_targets_raw[segment_start:segment_end]

                    # Set target to 1 if washing machine is active in the segment
                    sample_targets.append(1 if np.any(segment_data) else 0)

                # Add to our datasets
                self.samples.append(sample_features)
                self.targets.append(np.array(sample_targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]

        # Reshape sample to (channels, sequence_length) as expected by U-Time
        sample = sample.reshape(1, -1)

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target,
                                                                       dtype=torch.long)


class BalancedWashingMachineDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, segment_size, n_segments_per_sample=10,
                 undersample_ratio=1.0, random_seed=42):
        """
        Create a balanced dataset for washing machine detection using undersampling.

        Parameters:
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing the data
        segment_size : int
            Size of each segment for classification
        n_segments_per_sample : int
            Number of segments in each sample
        undersample_ratio : float
            Ratio of negative (inactive) to positive (active) samples.
            Default is 1.0 (equal number of active and inactive samples).
        random_seed : int
            Random seed for reproducibility
        """
        self.df = dataframe
        self.segment_size = segment_size
        self.n_segments_per_sample = n_segments_per_sample
        self.sample_length = segment_size * n_segments_per_sample
        self.undersample_ratio = undersample_ratio

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Will store samples where washing machine is active at least once
        active_samples = []
        active_targets = []

        # Will store samples where washing machine is never active
        inactive_samples = []
        inactive_targets = []

        # Process each house separately
        for house in self.df['house_number'].unique():
            house_df = self.df[self.df['house_number'] == house].sort_values('datetime')

            # Skip houses with too little data
            if len(house_df) < self.sample_length:
                continue

            # Extract features and targets
            features = house_df['Aggregate'].values.astype(np.float32)
            washing_machine = house_df['washing_machine'].astype(int).values

            # Normalize features for better training
            features = (features - features.mean()) / (features.std() + 1e-8)

            # Calculate how many complete samples we can create
            n_samples = len(features) // self.sample_length

            # Process each sample
            for i in range(n_samples):
                start_idx = i * self.sample_length
                end_idx = (i + 1) * self.sample_length

                sample_features = features[start_idx:end_idx]
                sample_targets_raw = washing_machine[start_idx:end_idx]

                # Create segment-level targets
                sample_targets = []
                for j in range(self.n_segments_per_sample):
                    segment_start = j * segment_size
                    segment_end = (j + 1) * segment_size
                    segment_data = sample_targets_raw[segment_start:segment_end]

                    # Set target to 1 if washing machine is active in the segment
                    sample_targets.append(1 if np.any(segment_data) else 0)

                sample_targets = np.array(sample_targets)

                # Check if this sample has any active segments
                if np.any(sample_targets):
                    active_samples.append(sample_features)
                    active_targets.append(sample_targets)
                else:
                    inactive_samples.append(sample_features)
                    inactive_targets.append(sample_targets)

        # Calculate how many inactive samples to keep
        n_active = len(active_samples)
        n_inactive_to_keep = int(n_active * self.undersample_ratio)

        print(
            f"Before undersampling: {len(active_samples)} active samples, {len(inactive_samples)} inactive samples")

        # Randomly select inactive samples if we have more than needed
        if len(inactive_samples) > n_inactive_to_keep:
            indices = np.random.choice(len(inactive_samples), n_inactive_to_keep,
                                       replace=False)
            inactive_samples = [inactive_samples[i] for i in indices]
            inactive_targets = [inactive_targets[i] for i in indices]

        # Combine active and inactive samples
        self.samples = active_samples + inactive_samples
        self.targets = active_targets + inactive_targets

        # Shuffle the combined dataset
        combined = list(zip(self.samples, self.targets))
        np.random.shuffle(combined)
        self.samples, self.targets = zip(*combined)

        print(
            f"After undersampling: {len(active_samples)} active samples, {len(inactive_samples)} inactive samples")
        print(f"Total samples: {len(self.samples)}")

        # Calculate class balance after undersampling
        total_segments = sum(len(target) for target in self.targets)
        active_segments = sum(np.sum(target) for target in self.targets)
        print(
            f"Segment-level class distribution: {active_segments}/{total_segments} active segments "
            f"({100 * active_segments / total_segments:.2f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]

        # Reshape sample to (channels, sequence_length) as expected by U-Time
        sample = sample.reshape(1, -1)

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target,
                                                                       dtype=torch.long)

# Create datasets
train_dataset = WashingMachineDataset(train_df, segment_size)
test_dataset = WashingMachineDataset(test_df, segment_size)

print(f"Training sequences: {len(train_dataset)}")
print(f"Testing sequences: {len(test_dataset)}")

# Check data shapes and class balance
sample_x, sample_y = train_dataset[0]
print(f"Sample X shape: {sample_x.shape}")
print(f"Sample Y shape: {sample_y.shape}")
print(f"Class balance in sample: {sample_y.sum().item()}/{len(sample_y)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 4. Create a lightweight U-Time model configuration
model = UTimeModel(
    in_channels=1,
    n_classes=2,
    segment_size=segment_size,
    init_filters=4,
    depth=2,
    kernel_size=3,
    dropout=0.2,
    learning_rate=1e-3,
    log_confusion_matrix=True,
    log_predictions=True,
    pool_sizes=[8, 4],
    up_kernel_sizes=[4, 8]
)

# 5. Configure training with early stopping
trainer = L.Trainer(
    max_epochs=5,  # Light config: reduced from default 100
    callbacks=[
        L.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        L.callbacks.ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.4f}',
                                    save_top_k=3, mode='min')
    ],
    logger=L.loggers.TensorBoardLogger("lightning_logs", name="washing_machine_detection"),
    log_every_n_steps=10
)

# 6. Train the model
print("Training model...")
trainer.fit(model, train_loader, test_loader)

# 7. Evaluate model on both train and test sets
print("Evaluating on training data...")
train_results = trainer.test(model, train_loader)

print("Evaluating on testing data...")
test_results = trainer.test(model, test_loader)

print("Training metrics:")
print(train_results)
print("\nTesting metrics:")
print(test_results)
