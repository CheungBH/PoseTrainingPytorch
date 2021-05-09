import argparse

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--expFolder', default='test', type=str,
                    help='Experiment folder')

"----------------------------- Model options -----------------------------"
parser.add_argument('--dataset', default='coco', type=str,
                    help='Dataset choice: mpii | coco')
parser.add_argument('--data_cfg', default="config/data_cfg/data_default.json", type=str,
                    help='Path of data cfg file')

"----------------------------- Model options -----------------------------"
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--model_cfg', default="config/model_cfg/default/cfg_resnet18.json", type=str,
                    help='Path of model cfg file')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='epsilon')
parser.add_argument('--lr_schedule', default="step", type=str,
                    help='interval')
parser.add_argument('--crit', default='MSE', type=str,
                    help='Criterion type')
parser.add_argument('--loss_weight', default=0, type=int,
                    help='Criterion type')
parser.add_argument('--freeze', default=0, type=float,
                    help='freeze backbone')
parser.add_argument('--freeze_bn', action='store_true',
                    help='freeze bn')
parser.add_argument('--optMethod', default='rmsprop', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')
parser.add_argument('--sparse_s', default=0, type=float,
                    help='sparse')
parser.add_argument('--patience', default=6, type=float,
                    help='epoch of lr decay')

"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=120, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--trainBatch', default=12, type=int,
                    help='Train-batch size')
parser.add_argument('--validBatch', default=12, type=int,
                    help='Valid-batch size')
parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')
parser.add_argument('--train_worker', default=5, type=int,
                    help='num worker of train')
parser.add_argument('--val_worker', default=1, type=int,
                    help='num worker of val')
parser.add_argument('--save_interval', default=20, type=int,
                    help='interval')

opt, _ = parser.parse_known_args()

