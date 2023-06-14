from pathlib import Path
import argparse

from tqdm import tqdm

import torch

from src.model import SemiSupervisedModule, ResnetMultiProj
from src.utils import get_config


def save_weights(input_path: str) -> None:
    """Saves weights from checkpoint to .pth file
    Finds all .ckpt files in input_path and saves weights to .pth file in same directory

    Args:
        input_path: path to directory containing .ckpt files
    """

    input_path = Path(input_path)

    # find all .ckpt files
    ckpt_files = list(input_path.glob('**/*.ckpt'))

    for ckpt_file in tqdm(ckpt_files):

        # find config file for each checkpoint
        config_file = list(ckpt_file.parent.parent.glob('*.yaml'))[0]
        config = get_config(config_file)

        encoder = ResnetMultiProj(**config['encoder']).backbone

        module = SemiSupervisedModule.load_from_checkpoint(checkpoint_path=ckpt_file,
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
    parser.add_argument('--in_path', '-i', help='Path to directory with checkpoints', required=True)
    args = parser.parse_args()

    save_weights(args.in_path)
