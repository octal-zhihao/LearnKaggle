# author:octal 
# time:2024/7/18
import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import DInterface
from model import MInterface
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import wandb

def train(args):
    data_module = DInterface(data_dir=args.data_dir,
                             batch_size=args.batch_size,
                             val_split=args.val_split,
                             augment=True)

    model = MInterface.load_from_checkpoint(args.model_checkpoint,
                                            input_dim=args.input_dim,
                                            lr=args.lr,
                                            dropout_rate=args.dropout_rate)

    # early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    checkpoint_callback = ModelCheckpoint(monitor="val_log_rmse", mode='min', save_top_k=1, verbose=True)

    # 初始化 wandb
    wandb.init(project='HousePrices', config=args)
    # 创建 WandbLogger
    wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=args.max_epochs, callbacks=[checkpoint_callback],
                      logger=wandb_logger)
    trainer.fit(model, data_module)


def predict(args):
    # Initialize DataModule and Model
    data_module = DInterface(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.setup(stage='test')
    model = MInterface.load_from_checkpoint(args.model_checkpoint,
                                            input_dim=args.input_dim,
                                            lr=args.lr,
                                            dropout_rate=args.dropout_rate)

    # Prepare Test Data and DataLoader
    test_data = pd.read_csv(f'{args.data_dir}/test.csv')
    test_loader = data_module.test_dataloader()

    model.eval()
    test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch)
            test_pred.extend(preds)

    submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': np.array(test_pred).ravel()})
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=245)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--model_checkpoint', type=str,
                        default='lightning_logs/kclj5p21/checkpoints/epoch=9-step=320.ckpt')

    args = parser.parse_args()
    if args.is_train:
        train(args)
    else:
        predict(args)