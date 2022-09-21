import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import random
import os
from PIL import Image

class Autoencoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3,16, kernel_size=3, stride=1,padding=1)
		self.conv2 = nn.Conv2d(16,32, kernel_size=3, stride=1,padding=1)
		self.conv3 = nn.Conv2d(32,64, kernel_size=3, stride=1,padding=1)
		self.conv4 = nn.Conv2d(64,128, kernel_size=3, stride=1,padding=1)
		self.conv5 = nn.Conv2d(128,256, kernel_size=3, stride=1,padding=1)

		self.deconv5 = nn.ConvTranspose2d(256,128,3,2,1,0,1,True,2)
		self.deconv4 = nn.ConvTranspose2d(128,64,3,2,1,0,1,True,2)
		self.deconv3 = nn.ConvTranspose2d(64,32,3,2,1,0,1,True,2)
		self.deconv2 = nn.ConvTranspose2d(32,16,3,2,1,0,1,True,2)
		self.deconv1 = nn.ConvTranspose2d(16,3,3,2,1,0,1,True,2)

		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(3,2,1)

	def encoder(self,x, shortcut):
		x =self.maxpool(self.relu(self.conv1(x)))
		x =self.maxpool(self.relu(self.conv2(x)))
		x =self.maxpool(self.relu(self.conv3(x)))
		if shortcut: return x
		x =self.maxpool(self.relu(self.conv4(x)))
		x =self.maxpool(self.relu(self.conv5(x)))
		return x

	def decoder(self,x, shortcut):
		if not shortcut:
			x = self.deconv5(x)
			x= self.deconv4(x)
		x=self.deconv3(x)
		x=self.deconv2(x)
		x=self.deconv1(x)
		return x
		

	def forward(self,x):
		# randomly only train the first and last few layers, skipping the middle
		# ths is not classic 'u-net' architecture, just simple encoder-decoder stack
		shortcut =  random.random()<0.5
		print(shortcut)
		latent=self.encoder(x,shortcut)
		out=self.decoder(latent,shortcut)
		return out
		

def check_ae_works():
	ae = Autoencoder()
	input = torch.randn(3,256,256)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)

class AddImageNoise(object):
	def __init__(self,amount):
		self.amount=amount
	def __call__(self,sample):
		rndlittle=((torch.rand( sample.shape)-0.5)*self.amount*128).int()
		resize=transforms.Resize((sample.shape[1],sample.shape[2]))
		rndbig=resize((torch.rand( sample.shape[0],int(sample.shape[1]/2),int(sample.shape[2]/2)) -0.5)*self.amount*255).int()
		rnd=rndlittle+rndbig

		print(torch.max(sample),torch.min(sample))
		print(torch.max(rnd),torch.min(rnd))

		sample=torch.add(sample,rnd)
#		sample = sample+((rnd-0.5)*self.amount)
#		sample=torch.clip(sample,0,255)
#		sample = sample+64

		print(torch.max(sample),torch.min(sample))
		sample=torch.clip(sample,0,255)
		print(torch.max(sample),torch.min(sample))
		return sample.float()*(1.0/255.0)
		

class NoisedImageDataset(Dataset):
	def __init__(self,dirname):
		print("init dataset from dir: ",dirname)
		self.images=[]
		
		for i,x in enumerate(os.listdir(dirname)):
			img=Image.open(dirname+x)
			imgarr=numpy.array(img)
			imgarr=torch.tensor(imgarr.transpose((2,0,1))) # numpy HWC -> torch CHW
			if i<32:
				print("img[",i,"] ",x," size=",imgarr.shape)
			self.images.append(imgarr)
		print("total images=",len(self.images))

		self.add_noise=AddImageNoise(0.5)
			

	def __len__(self):
		return len(self.images)

	def __getitem__(self,idx):
		img=self.images[idx];
		return {'input': self.add_noise(img), 'output':img}
		
		

def make_dataloader(dirname="../training_images/"):



	dataset = NoisedImageDataset(dirname)
	for x in range(0,3):
		foo=dataset[x]
		transforms.ToPILImage()(foo['input']).show()
	return DataLoader(
		dataset,
		batchsize=16, 
		shuffle=true)
		
	

def main():
	print("initializing device..")

	if torch.cuda.is_available(): device=torch.device("cuda")
	elif torch.backends.mps.is_available(): device = torch.device("mps")
	else: device = torch.device("cpu")

	print("grabbing dataset..")
	dataloader=make_dataloader("../training_images/")


	print("training on some images.")
	torch.save(ae.state_dict(), "my_trained_ae")

main()

		
