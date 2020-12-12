#!/bin/bash

#model='DnQ'
#dataset='RotNIST'
#test_model_name='./data/saved_models/dnq_cfg_fb_mnist.tar'

#model='VGG19'
#dataset='RotNIST'
#test_model_name='./data/saved_models/vgg19_cfg_a_mnist.tar'

#model='ResNet50'
#dataset='RotMNIST'
#test_model_name='./data/saved_models/resnet50_cfg_a_mnist.tar'

#model='DnQ'
#dataset='CIFAR10'
#test_model_name='./data/saved_models/dnq_cfg_fb_cifar10.tar'

#model='VGG19'
#dataset='CIFAR10'
#test_model_name='./data/saved_models/vgg19_cfg_a_cifar10.tar'

model='ResNet50'
dataset='CIFAR10'
test_model_name='./data/saved_models/resnet50_cfg_a_cifar10.tar'

SET=$(seq 0 359)

for i in $SET 
do
  python train.py --model $model --test_dataset $dataset --test_model_name $test_model_name --single_rotation_angle $i --device 'cuda:1'
done
