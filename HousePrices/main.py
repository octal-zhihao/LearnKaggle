# author:octal 
# time:2024/7/18

import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data import DInterface
from model import MInterface
from pytorch_lightning.loggers import WandbLogger
import wandb

def main(args):
    data_module = DInterface(data_dir=args.data_dir, batch_size=args.batch_size, val_split=args.val_split)
    model = MInterface(input_dim=args.input_dim, lr=args.lr, num_heads=args.num_heads, dropout_rate=args.dropout_rate)

    early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, verbose=True)

    # 初始化 wandb
    wandb.init(project='HousePrices', entity='octal-zhihao-zhou')
    # 创建 WandbLogger
    wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=args.max_epochs, callbacks=[early_stopping_callback, checkpoint_callback],
                      logger=wandb_logger)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=79)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    args = parser.parse_args()
    main(args)
