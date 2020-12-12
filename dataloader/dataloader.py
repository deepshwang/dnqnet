from dataloader.dataset import RotNISTDataset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler


def RotNISTDataloader(args, mode, T):
	dataset = RotNISTDataset(args, mode, T)

	if mode == 'train':
		batch_size = args.train_batch
	elif mode == 'test' or mode == 'val':
		batch_size = args.test_batch


	dataloader = DataLoader(dataset=dataset,
							num_workers=args.num_workers,
							batch_size=batch_size,
							shuffle=True)

	return dataloader

def MNISTDataloader(args, mode, T):

	# Set batch-size
	if mode == 'train':
		batch_size = args.train_batch
	elif mode == 'test' or mode == 'val':
		batch_size = args.test_batch

	# Load dataset
	dataset = datasets.MNIST(root='./data', train=(mode == 'train' or mode =='val'),
							 download=True, transform=T)

	indices = list(range(len(dataset)))
	split = int(args.train_val_ratio*len(dataset))

	if mode == 'train':
		sampler = SubsetRandomSampler(indices[:split])
		shuffle = False
	elif mode == 'val':
		sampler = SubsetRandomSampler(indices[split:])
		shuffle = False
	else:
		sampler = None
		shuffle = False

	dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
							sampler=sampler,
							num_workers=args.num_workers,
							shuffle=shuffle)

	return dataloader

def CIFAR10Dataloader(args, mode, T):

	# Set batch-size
	if mode == 'train':
		batch_size = args.train_batch
	elif mode == 'test' or mode == 'val':
		batch_size = args.test_batch

	# Load dataset
	dataset = datasets.CIFAR10(root='./data', train=(mode == 'train' or mode =='val'),
							 download=True, transform=T)

	indices = list(range(len(dataset)))
	split = int(args.train_val_ratio*len(dataset))

	if mode == 'train':
		sampler = SubsetRandomSampler(indices[:split])
		shuffle = False
	elif mode == 'val':
		sampler = SubsetRandomSampler(indices[split:])
		shuffle = False
	else:
		sampler = None
		shuffle = True

	dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
							sampler=sampler,
							num_workers=args.num_workers,
							shuffle=shuffle)

	return dataloader

def CIFAR100Dataloader(args, mode, T):

	# Set batch-size
	if mode == 'train':
		batch_size = args.train_batch
	elif mode == 'test' or mode == 'val':
		batch_size = args.test_batch

	# Load dataset
	dataset = datasets.CIFAR100(root='./data', train=(mode == 'train' or mode =='val'),
							 download=True, transform=T)

	indices = list(range(len(dataset)))
	split = int(args.train_val_ratio*len(dataset))

	if mode == 'train':
		sampler = SubsetRandomSampler(indices[:split])
		shuffle = False
	elif mode == 'val':
		sampler = SubsetRandomSampler(indices[split:])
		shuffle = False
	else:
		sampler = None
		shuffle = True

	dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
							sampler=sampler,
							num_workers=args.num_workers,
							shuffle=shuffle)

	return dataloader

