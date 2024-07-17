import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from data import DInterface
from model import MInterface


def main(args):
    data_module = DInterface(data_dir=args.data_dir, batch_size=args.batch_size, val_split=args.val_split)
    model = MInterface(input_dim=args.input_dim, lr=args.lr)

    early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=15)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, verbose=True)
    csv_logger = CSVLogger(save_dir='logs/', name='titanic')

    trainer = Trainer(max_epochs=args.max_epochs, callbacks=[early_stopping_callback, checkpoint_callback],
                      logger=csv_logger)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()
    main(args)
