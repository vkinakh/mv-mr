from pathlib import Path
import argparse

from tqdm import tqdm

import torch

from src.model import CLIPSelfSupervisedModule
from src.utils import get_config


def save_encoder_weights(input_path):

    input_path = Path(input_path)

    # find all .ckpt files
    ckpt_files = list(input_path.glob('**/*.ckpt'))

    for ckpt_file in tqdm(ckpt_files):

        # find config file for each checkpoint
        config_file = list(ckpt_file.parent.parent.glob('*.yaml'))[0]
        config = get_config(config_file)

        # load module
        module = CLIPSelfSupervisedModule.load_from_checkpoint(checkpoint_path=ckpt_file,
                                                               config=config)
        encoder = module.encoder.eval()
        finetuner = module.online_finetuner.eval()

        # save encoder weights
        output_path = ckpt_file.parent / f'{ckpt_file.stem}.pth'
        print(f'Saving encoder weights to {output_path}')
        torch.save(
            {
                'encoder': encoder.state_dict(),
                'online_finetuner': finetuner.state_dict(),
            },
            output_path
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', '-i', help='Path to checkpoint', required=True)
    args = parser.parse_args()

    save_encoder_weights(args.in_path)
