# Detecting Deepfakes using Benford's Law

This is an implementation of the paper titled "On the use of Benford's law to detect GAN-generated images" ([PDF](https://arxiv.org/abs/2004.07682)). Training and testing is done using the DoGANs dataset ([link](https://grip-unina.github.io/DoGANs/))

## A Quick Primer

Benford's Law is a power-law probability distribution based on occurence of leading digits in natural number sets. Specifically, the law predicts that:
- The number 1 appears as the leading digit about 30% of the time
- The number 2 appears as the leading digit about 17.6% of the time
- The number 3 appears as the leading digit about 12.5% of the time
- ...and so on, with 9 appearing as the leading digit only about 4.6% of the time

This distribution applies to a wide variety of datasets, including financial transactions, where it is one of the tools used by auditors to detect fraud.

In the case of images, Benford's Law can be used to detect GAN-generated images by analyzing the leading digits of the DCT coefficients of the images, by training a Random Forest Classifier on the Benford features described in the original paper (linked above).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-detection-benfords-law.git
   cd deepfake-detection-benfords-law
   ```

2. Install the required dependencies using `poetry`:
   ```bash
   poetry install
   ```

## Usage

### Dataset Preparation

1. Download the DoGANs dataset from [here](https://grip-unina.github.io/DoGANs/).
2. Extract the dataset to a directory of your choice.
3. Update the `data_dir` path in the `config.yml` file to point to your dataset directory.


### Configuration

Before running the training or inference scripts, make sure to review and update the `config.yml` file with your desired settings:

```yaml
data:
data_dir: /path/to/dogans/dataset
train_split: 0.8
val_split: 0.1
test_split: 0.1
model:
type: benford_classifier
```

### Training
```bash
python main.py train -c config.yml -o output_dir
```

- `-c`: Path to the configuration file
- `-o`: Output directory for saving model checkpoints and logs

### Inference

```bash
python main.py predict -i image.png -m model.pkl
```

- `-i`: Path to the input image
- `-m`: Path to the trained model file
