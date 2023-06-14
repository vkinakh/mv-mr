import argparse
from pathlib import Path

import torch

from src.model import ResnetMultiProj, VocLinearEvalModule
from src.utils import get_config


def save_weights(input_path):

    input_path = Path(input_path)

    # find all .ckpt files
    ckpt_files = list(input_path.glob('**/*.ckpt'))

    for ckpt_file in ckpt_files:

        # find config file for each checkpoint
        config_file = list(ckpt_file.parent.parent.glob('*.yaml'))[0]
        config = get_config(config_file)

        encoder = ResnetMultiProj(**config['encoder']).backbone

        module = VocLinearEvalModule.load_from_checkpoint(checkpoint_path=ckpt_file,
                                                          config=config, encoder=encoder)
        encoder = module.encoder.eval()
        classifier = module.classifier.eval()

        # save encoder weights
        output_path = ckpt_file.parent / f'{ckpt_file.stem}.pth'
        print(f'Saving encoder weights to {output_path}')
        torch.save(
            {
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
            },
            output_path
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', '-i', help='Path to checkpoint', required=True)

    args = parser.parse_args()
    save_weights(args.in_path)
