import pandas as pd
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from data import DInterface

from model import MInterface


def main(args):
    # Load data module
    data_module = DInterface(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.setup(stage='test')

    # Load model
    model = MInterface.load_from_checkpoint(checkpoint_path=args.checkpoint_path, input_dim=args.input_dim)

    # Create trainer
    trainer = pl.Trainer()

    # Predict
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())
    predictions = torch.cat(predictions).numpy()

    # Load test.csv to get Id column
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")

    # Create submission dataframe
    submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': predictions.flatten()})

    # Save submission
    submission_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='submission.csv')
    parser.add_argument('--input_dim', type=int, default=79)  # Assuming 79 features after preprocessing
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    main(args)
