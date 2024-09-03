from benford_gan.core import Label
from benford_gan.model import (
    load_config_and_validate,
    label_training_images,
    BenfordClassifier
)
from benford_gan.helper import load_image_from_file


import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, help="Train or inference command", choices=['train', 'predict'])
    # parser.add_argument('path', type=str, help="For training, a *.yaml config file, for inference, an image path.")
    parser.add_argument('--cfg', '-c', type=str, help="Config.yml filepath for training config")
    parser.add_argument('--image', '-i', type=str, help="Image filepath for classifier inference.")
    parser.add_argument('--model', '-m', type=str, help="Trained Benford classifier model filepath.")
    
    args = parser.parse_args()

    print(args)
    # print(args.__dict__)
    # print(args.cfg)
    if args.cmd.lower() == 'train':
        cfg = load_config_and_validate(args.cfg)
        classifier = BenfordClassifier(cfg)

        training_imgs = []
        for folder in cfg.natural_dirs:
            training_imgs.extend(label_training_images(folder), Label.NATURAL)
        for folder in cfg.deepfake_dirs:
            training_imgs.extend(label_training_images(folder), Label.GAN_GENERATED)
        
        classifier.train(training_imgs, cfg.output_dir)

    elif args.cmd.lower() == 'predict':
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"{args.image} does not exist.")
        if not os.path.isfile(args.model):
            raise FileNotFoundError(f"{args.model} does not exist")
        
        classifier = BenfordClassifier.load(args.model)
        img = load_image_from_file(args.image)
        
        result = classifier.predict(img)[0]
        print(f"Result: {Label(result)}")




