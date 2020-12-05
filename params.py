import argparse
import torch

###### PATH VARIABLES #####

ROTNIST_ROOT = "./data/RotNIST/data"



###### MODEL CONFIGURATIONS #####

MODEL_CFGS={'A': {'localcfg': [16, 16],
				  'globalcfg' : [16, 16, 32, 'P', 32, 32, 64, 'P', 64, 64, 128, 'P', 128, 128, 256, 'P'],
				  'clscfg': [512, 10]},
			'B': {'localcfg': [16, 16],
				  'globalcfg' : [16, 16, 32, 'P', 32, 32, 64, 'P', 64, 64, 128, 'P', 128, 128, 256, 'P'],
				  'clscfg': [512, 256, 10]},
			'C': {'localcfg': [16, 16, 32, 32],
				  'globalcfg' : [32, 64, 'P', 64, 128, 'P', 128, 256, 'P', 256, 256, 'P'],
				  'clscfg': [128, 10]},
			'D': {'localcfg': [16, 32, 32],
				  'globalcfg' : [32, 64, 'P', 64, 128, 'P', 128, 256, 'P', 256, 256, 'P'],
				  'clscfg': [128, 10]},
			}

MODEL_CFGS_V3={'A': [[1, 8], [8, 8, 16], [16, 16, 16], 'A',
					 [16, 32, 32], [32, 32, 64], 'A',
					 [64, 64, 128], [128, 128, 128], 'A', 'A'],
			   'B': [[1, 8], [8, 8, 16], 'A', [16, 16, 16],
					 [16, 32, 32], 'A', [32, 32, 64],
					 [64, 64, 128], 'A', [128, 128, 128], 'A', [128, 128, 128]],
			   'C': [[3, 16, 16], [16, 16, 32, 3]],
			   'D': [[1, 4], [4, 8, 8], [8, 8, 16], [16, 16, 32], [32, 32, 32], [32, 32, 64], [64, 64, 128]],
			   'E': [[3, 8], [8, 8, 16], [16, 16, 32], [32, 32, 64],
					 [64, 64, 128], [128, 128, 256], [256, 256, 512]],

			   # Similar to VGG 16
			   'F': [[3, 64, 64], [64, 64, 64], [64, 128, 128], [128, 128, 128],
						 [128, 128, 128], [128, 256, 256], [256, 256, 256], [256, 256, 256], [256, 256, 256],
						 [256, 512, 512], [512, 512, 512], [512, 512, 512], [512, 512, 512]],

			   'G': [[3, 64, 64], [64, 64, 64], [64, 64, 64], [64, 64, 128],
						 [128, 128, 128], [128, 128, 128], [128, 128, 128], [128, 256, 256], [256, 256, 256],
						 [256, 256, 256], [256, 256, 256], [256, 256, 256], [256, 256, 256], [256, 512, 512],
						 [512, 512, 512], [512, 512, 512]]
			   }

CLASSIFIER_CFGS = {'A': [512, 256, 100],
				   'B': [512, 256, 10]}

CLASSIFIER_CFGS_VGG19 = {'A': [512, 256, 10],
				   'B': [512, 256, 10]}

CLASSIFIER_CFGS_ResNet50 = {'A': [2048, 256, 10]}

##### ARGUMENT PARSER #####

def argument_parser():
	parser = argparse.ArgumentParser(description='Argument parser for DnQNet training/testing')


	parser.add_argument('--model', type=str,
						default='DnQ',
						choices=['DnQ', 'ResidualDnQ', 'CNN', 'VGG19', 'ResNet50', 'C8SteerableCNN', 'C16SteerableCNN'],
						help='type of model to train')


	parser.add_argument('--train_dataset', type=str,
						default='CIFAR10',
						choices=['MNIST', 'RotNIST', 'CIFAR10', 'RotCIFAR10', 'CIFAR100', 'RotCIFAR100'],
						help='Dataset to train')

	parser.add_argument('--test_dataset', type=str,
						default='RotCIFAR10',
						choices=['MNIST', 'RotNIST', 'CIFAR10', 'RotCIFAR10', 'CIFAR100', 'RotCIFAR100'],
						help='Dataset to test')

	parser.add_argument('--single_rotation_angle', type=int,
						default=0,
						help='Single rotation angel to test')

	parser.add_argument('--tsne_model', type=str,
						default='VGG19',
						choices=['DnQ', 'ResidualDnQ', 'CNN', 'VGG19', 'ResNet50'],
						help='type of model to visualize via tsne')


	parser.add_argument('--tsne_dataset', type=str,
						default='CIFAR10',
						choices=['MNIST', 'CIFAR10', 'CIFAR100', 'RotCIFAR100'],
						help='Dataset to visualize via tsne')

	parser.add_argument('--test_model_name', type=str,
					default='./data/saved_models/dnq_cfg_fb_cifar10.tar',
					# default='./data/saved_models/vgg19_cfg_a_cifar10.tar',
					# default='./data/saved_models/resnet50_cfg_a_cifar10.tar',
					# default='./data/saved_models/vgg19_cfg_a_mnist.tar',
					# default='./data/saved_models/dnq_cfg_fb_mnist.tar',
					# default='./data/saved_models/resnet50_cfg_a_mnist.tar',
					# default='./data/saved_models/checkpoint.pth.tar',
					help='testing model state dict')

	parser.add_argument('--tsne_state_dict_path', type=str,
					# default='./data/saved_models/dnq_cfg_fb_cifar10.tar',
					default='./data/saved_models/vgg19_cfg_a_cifar10.tar',
					# default='./data/saved_models/resnet50_cfg_a_cifar10.tar',
					# default='./data/saved_models/vgg19_cfg_a_mnist.tar',
					# default='./data/saved_models/dnq_cfg_fb_mnist.tar',
					# default='./data/saved_models/checkpoint.pth.tar',
					help='visualizing model state dict')



	## DnQ V1 Network Arguments

	parser.add_argument('--in_channels', type=int,
						default=3,
						choices=[1, 3, 5],
						help='JUST FOR DNQ: 3 for grayscale input and 5 for rgb input')

	parser.add_argument('--in_classifier_len', type=int, 
						default=256,
						help='input length of classifier')

	parser.add_argument('--p', type=int, 
					default=3,
					help='power for Geometric Mean Pooling')

	parser.add_argument('--pool_size', type=int,
					default=2,
					help='pool size Geometric Mean Pooling')

	parser.add_argument('--pool_type', type=str,
						default='GeM',
						choices=['GeM', 'M'],
						help='type of model to train')

	parser.add_argument('--max_pool_size', type=int,
					default=2,
					help='maximum pool size to yield 1 by 1')

	## DnQNet v2 Arguments

	parser.add_argument('--backbone', type=str,
						default='MNISTCNN',
						choices=['VGG16', 'ResNet50', 'MNISTCNN'],
						help='type of backbone for DnQv2')

	## DnQNet v3 Arguments
	parser.add_argument('--edge_neighbor', type=int,
						default=3,
						help='Set diameter for fixed radius near-neighbors search (kernel size)')

	parser.add_argument('--graph_agg_type', type=str,
						default='A',
						choices=['M', 'A'], # 'A' is conceptually right for graphconv / G is gaussian
						help='Edge information aggregation type')

	parser.add_argument('--point_agg_type', type=str,
						default='A',
						choices=['M', 'A'],
						help='Point-net like vertex aggregation type')

	parser.add_argument('--loss_function', type=str,
						default='ce',
						choices=['ce', 'arcface', 'sphereface', 'cosface'],
						help='Point-net like vertex aggregation type')

	parser.add_argument('--optimizer', type=str,
						default='adam',
						choices=['adam', 'sgd'],
						help='optimizer type')


	## Train / Test Arguments

	parser.add_argument('--test_only', type=bool,
						default=True,
						help='Conduct testing only')

	parser.add_argument('--train_batch', type=int, 
						default=64,
						help='Train batch size')

	parser.add_argument('--test_batch', type=int, 
						default=64,
						help='Test & validation batch size')

	parser.add_argument('--epochs', type=int, 
						default=1000,
						help='Number of epochs for training')
	parser.add_argument('--num_workers', type=int,
						default=1,
						help='Number of workers')

	parser.add_argument('--batch_iter', type=int, 
						default=100,
						help='batch iteration size for logging')

	parser.add_argument('--device', type=str, 
						default='cuda' if torch.cuda.is_available() else 'cpu',
						help='Test batch size')


	parser.add_argument('--train_val_ratio', type=float,
					default=0.8,
					help='percentage of train data over the whole train dataset')

	parser.add_argument('--save_bestmodel_name', type=str,
					default='./data/saved_models/checkpoint.pth.tar',
					help='percentage of train data over the whole train dataset')



	parser.add_argument('--save_mode', type=str,
					default='params_only',
					choices=['params_only', 'entire_models', ""],
					help='params: saves only the model parameters - load_state_dict' \
						 'entire_models: saves the entire model')

	parser.add_argument('--resume_training', type=bool,
						default=False,
						help='Whether to resume training from checkpoint')

	## Visualization arguments
	parser.add_argument('--viz_network', type=str,
					default='DnQ',
					choices=['DnQ', 'CNN', 'E2CNN'],
					help='percentage of train data over the whole train dataset')



	args = parser.parse_args()
	return args
