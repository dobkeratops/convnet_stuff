Testbed for writing convnet inference in opencl

plus PyTorch program to experiment with training u-nets and more

goal is to use neural nets to augment retro game/indie game art, with a small/custom lib for integration

current training experiments in pytorch/...

cuda version & cpu version out of sync , cleanpu incoming

during training , throws image showing progress in the current working directory, and writes it to /var/www/html 
which some linux distros seem to serve out of the box
\Should look something like this, showinng the grouped input and 
outputs(aka 'target') + 
current network generated output
![](screenshot.jpeg)

## TRAINING (WIP)

use pytorch/trainer.py   *

the idea is to take multiple image planes which could be previous/next 
frames 
of video to predict the next, RGB+material markup, 

example invocation:
```
python3 pytorch/trainer.py -i my_image_dir   -o my_trained_model_dir    -k 5  -f 24 -z 256 -l 5 -s
```

`-i`	: directory holding source images

`-k 5` 	: convolutional filter kernels, 5x5

`-f 24`	: 1st (and final) layer of features

`-z 256`	: latent space feature depth 

`-l 5`	: 5 layers (24,48,96,192,256) each downsampled but deeper

`-s`	:enable Skip Connections, "u-net" style (default is off, it will use a single shortcut output from the middle instead) 

`-o my_trained_model_dir`	
	: where it periodically saves the trained models, 
		.pty format state_dict files

this expects:
example input&outputs in `my_image_dir/`, according to this naming 
convention:

```
imagename1_INPUT0.JPG  , imagename1_INPUT1.JPG  , imagename1_OUTPUT.JPG
anotherimage_INPUT0.JPG  , anotherimage_INPUT1.JPG  , anotherimage_OUTPUT.JPG
```
etc to train *_INPUT0, *_INPUT1 -> *_OUTPUT ( 6 -> 3 channels total
(rgb0,rgb1)->rgb

it can alsoo train single input -> multiple outputs,
e.g.
```
foo_INPUT.JPG  ->  foo_OUTPUT0.JPG  foo_OUTPUT1.JPG
bar_INPUT.JPG ->  bar_OUTPUT0.JPG bar_OUTPUT1.JPG 
```
etc

multi in -> multi out is untested

(* it started life as an autoencoder and may stil get a feature to 
autencode more inputs, with a different decoder branch?)

It uses a single encoder and decoder to take in and generate all the 
channels, essentially treating it as a deeper image,they must have some 
kind of correlation (i.e. multiple view inputs would not work yet, this is 
on the TODO list..)

Watch training progress on localhost/training_progress.html  , tested on 
ubuntu which serves var/www/html . if it doeesn't find this it will write 
training_progress.jpg screenshots

## RUNTIME

early days, there's a simple implementation of convolutional nets in 
OpenCL running in an SDL window in the cpp file, but it doesn't load the 
trained models yet
