from benford_gan import (
    helper,
    core,
    config
)

from dataclasses import dataclass
from math import ceil
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tqdm
from typing import NamedTuple
import yaml


@dataclass
class BenfordClassifierConfig:
    natural_dir: str
    deepfake_dir: str
    output_dir: str
    freq: list[int]
    bases: int
    qtables: list[config.QTable]
    batch_size: int

    def __repr__(self) -> str:
        # qtable_str = " \n".join([qt.name for qt in self.qtables])

        return f"Benford GAN Training Config:\n"\
                f"  Natural image directory: {self.natural_dir}\n"\
                f"  Deepfake image directory: {self.deepfake_dir}\n"\
                f"  DCT frequencies: {self.freq}\n"\
                f"  Radix: {self.base}\n"\
                f"  Quantization tables: {', '.join([qt.name for qt in self.qtables])}\n"\
                f"  Batch size: {self.batch_size}"


class TrainingImage(NamedTuple):
    filepath: str
    label: core.Label

    def __repr__(self) -> str:
        return f"Training Image: {self.filepath} ({self.label.name})"


class BenfordClassifier:

    def __init__(self, cfg: BenfordClassifierConfig) -> None:
        self.model = RandomForestClassifier(warm_start=True)
        self.bases = cfg.bases
        self.frequencies = cfg.freq
        self.quantization_tables = cfg.qtables
        self.training_batch_size = cfg.batch_size

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def train(self, imgs: list[TrainingImage], output_dir):
        features = process_training_images(
            training_imgs=imgs,
            frequencies=self.frequencies,
            bases=self.bases,
            qtables=self.quantization_tables,
            batch_size=self.training_batch_size,
            output_dir=output_dir
        )
        train, test = train_test_split(features, test_size = 0.3, train_size = 0.7)
        train_data = []
        train_labels = []
        for data in train:
            train_data.append(data.feature)
            train_labels.append(data.label)
        self.model.fit(train_data, train_labels)

        test_data = []
        test_labels = []
        for data in test:
            test_data.append(data.feature)
            test_labels.append(data.label)
        x = self.model.predict(test_data)

    def predict(img: np.ndarray):
        pass


def load_config_and_validate(cfg_path: str):
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError("The provided config path is invalid.")
    with open(cfg_path) as f:
        cfg_dict = yaml.load(stream=f, Loader=yaml.FullLoader)

    if not os.path.isdir(cfg_dict['natural_dir']):
        raise NotADirectoryError
    if not os.path.isdir(cfg_dict['deepfake_dir']):
        raise NotADirectoryError
    if not os.path.isdir(cfg_dict['output_dir']):
        raise NotADirectoryError
    if not cfg_dict['freq_count'] > 0 and cfg_dict['freq_count'] < 64:
        raise ValueError("Incorrect DCT frequency values. Ensure that 0 < f < 64 for all entries.")
    if not all(base > 0 for base in cfg_dict['bases']):
        raise ValueError(f"Radix entries cannot be 0 or negative (set value: {cfg_dict['bases']}).")
    if not cfg_dict['batch_size'] > 0:
        raise ValueError(f"Batch size cannot be 0 or negative (set value: {cfg_dict['batch_size']}).")
    
    cfg = BenfordClassifierConfig(
        natural_dir=cfg_dict['natural_dir'],
        deepfake_dir=cfg_dict['deepfake_dir'],
        output_dir=cfg_dict['output_dir'],
        freq=[f for f in range(cfg_dict['freq_count'])],
        bases=cfg_dict['bases'],
        qtables=[config.QTable[name.upper()] for name in cfg_dict['qtables']],
        batch_size=cfg_dict['batch_size']
    )

    return cfg


def load_training_images(nat_dir, gan_dir):
    nat_files = os.listdir(os.path.abspath(nat_dir))
    gan_files = os.listdir(os.path.abspath(gan_dir))

    training_imgs = [
        TrainingImage(
            os.path.join(nat_dir, file), core.Label.NATURAL
        ) for file in nat_files
    ]
    training_imgs.extend([
        TrainingImage(
            os.path.join(gan_dir, file), core.Label.GAN_GENERATED
        ) for file in gan_files
    ])

    return training_imgs


def generate_benford_features_batch(
    file_batch: list[TrainingImage],
    freq: list[int],
    bases: list[int],
    qtables: list
) -> list[core.BenfordFeature]:
    features = []
    for file in file_batch:
        img = helper.load_image_from_file(file.filepath)
        feature = core.generate_benford_feature(img, freq, bases, qtables, file.label.value)
        features.append(feature)
    return features


def save_batch_features(features: list[core.BenfordFeature], target_dir: str, batch_num: int, batch_count: int):
    if not os.path.isdir(target_dir):
        raise NotADirectoryError
    path = os.path.join(target_dir, f"benford_features_{batch_num}_of_{batch_count}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(features, f)


def process_training_images(
    training_imgs: list[TrainingImage],
    frequencies,
    bases,
    qtables,
    batch_size,
    output_dir
) -> list[core.BenfordFeature]:
    features = []
    # TODO: checkpointing, resume feature generation
    batch_count = ceil(len(training_imgs) / batch_size)
    for i, batch in tqdm.tqdm(helper.enumerated_batch_generator(training_imgs, batch_size)):
        batch_features = generate_benford_features_batch(batch, frequencies, bases, qtables)
        save_batch_features(batch_features, output_dir, i, batch_count)
        features.extend(batch_features)
    return features


def collect_training_images(cfg):
    nat_files = os.listdir(os.path.abspath(cfg.natural_dir))
    gan_files = os.listdir(os.path.abspath(cfg.deepfake_dir))

    training_imgs = [
        TrainingImage(
            os.path.join(cfg.natural_dir, file), core.Label.NATURAL
        ) for file in nat_files
    ]
    training_imgs.extend([
        TrainingImage(
            os.path.join(cfg.deepfake_dir, file), core.Label.GAN_GENERATED
        ) for file in gan_files
    ])
    return training_imgs
