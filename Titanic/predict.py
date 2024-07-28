import argparse
import pandas as pd
import numpy as np
import torch
from datasets import DInterface
from model import MInterface

def predict(args):
    data_module = DInterface(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.setup(stage='test')


    model = MInterface.load_from_checkpoint(args.model_checkpoint, input_dim=args.input_dim, lr=args.lr, num_heads=args.num_heads)

    # 创建测试数据集和数据加载器
    test_data = pd.read_csv(f'{args.data_dir}/test.csv')
    test_loader = data_module.test_dataloader()

    model.eval()
    test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch)
            preds = (preds.numpy() > 0.5).astype(int)
            test_pred.extend(preds)

    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': np.array(test_pred).ravel()})
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_checkpoint', type=str, default='lightning_logs/8ic1l4fh/checkpoints/epoch=64-step=2925.ckpt', help='Path to the model checkpoint for prediction')
    parser.add_argument('--input_dim', type=int, default=8, help='Input dimension for the model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate for the model')
    args = parser.parse_args()
    predict(args)
