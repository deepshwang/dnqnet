#!/bin/bash

#model='DnQ'
#dataset='RotNIST'
#test_model_name='./data/saved_models/dnq_cfg_fb_mnist.tar'

#model='VGG19'
#dataset='RotNIST'
#test_model_name='./data/saved_models/vgg19_cfg_a_mnist.tar'

#model='ResNet50'
#dataset='RotNIST'
#test_model_name='./data/saved_models/resnet50_cfg_a_mnist.tar'

#model='e2cnn-c8'
#dataset='RotNIST'
#test_model_name='./data/saved_models/c8_mnist.tar'

#model='e2cnn-c16'
#dataset='RotNIST'
#test_model_name='./data/saved_models/c16_mnist.tar'


#model='DnQ'
#dataset='RotCIFAR10'
#test_model_name='./data/saved_models/dnq_cfg_fb_cifar10.tar'

#model='VGG19'
#dataset='RotCIFAR10'
#test_model_name='./data/saved_models/vgg19_cfg_a_cifar10.tar'

#model='ResNet50'
#dataset='RotCIFAR10'
#test_model_name='./data/saved_models/resnet50_cfg_a_cifar10.tar'

#model='e2cnn-c8'
#dataset='RotCIFAR10'
#test_model_name='./data/saved_models/c8_cifar10.tar'

model='e2cnn-c16'
dataset='RotCIFAR10'
test_model_name='./data/saved_models/c16_cifar10.tar'

SET=$(seq 0 24)

for i in $SET 
do
  j=$((i * 15))
  python train.py --model $model --test_dataset $dataset --test_model_name $test_model_name --single_rotation_angle $j --device 'cuda:1'
done
