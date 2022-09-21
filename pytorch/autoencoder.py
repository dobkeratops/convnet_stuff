import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import psutil

import random
import os
from PIL import Image

def debug(*a):
	#print(*a)
	pass

class Autoencoder(nn.Module):
	def __init__(self):
		self.shown=None
		self.iter=0
		self.nextshow=100
		super().__init__()
		self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=1,padding=1)
		self.shrink1 = nn.Conv2d(16,16, kernel_size=2, stride=2,padding=0)
		self.conv2 = nn.Conv2d(16,24, kernel_size=3, stride=1,padding=1)
		self.shrink2 = nn.Conv2d(24,24, kernel_size=2, stride=2,padding=0)

		self.conv3 = nn.Conv2d(24,32, kernel_size=3, stride=1,padding=1)
		self.shrink3 = nn.Conv2d(32,32, kernel_size=2, stride=2,padding=0)

		self.conv4 = nn.Conv2d(32,48, kernel_size=3, stride=1,padding=1)
		self.shrink4 = nn.Conv2d(48,48, kernel_size=2, stride=2,padding=0)

		self.conv5 = nn.Conv2d(48,64, kernel_size=3, stride=1,padding=1)
		self.shrink5 = nn.Conv2d(64,64, kernel_size=2, stride=2,padding=0)

		


		inpad=0;outpad=0;groups=1

		self.deconv5 = nn.ConvTranspose2d(64,64, 2,2,inpad,outpad,groups,True,2)
		self.conv5up = nn.Conv2d(64,48, kernel_size=3, stride=1,padding=1)

		self.deconv4 = nn.ConvTranspose2d(48,48, 2,2,inpad,outpad,groups,True,2)
		self.conv4up = nn.Conv2d(48,32, kernel_size=3, stride=1,padding=1)

		
		self.deconv3 = nn.ConvTranspose2d(32,32, 2,2,inpad,outpad,groups,True,2)
		self.conv3up = nn.Conv2d(32,24, kernel_size=3, stride=1,padding=1)

		self.deconv2 = nn.ConvTranspose2d(24,24, 2,2,inpad,outpad,groups,True,2)
		self.conv2up = nn.Conv2d(24,16, kernel_size=3, stride=1,padding=1)

		self.deconv1 = nn.ConvTranspose2d(16,16,	2,2,inpad,outpad,groups,True,2)
		self.conv1up = nn.Conv2d(16,3, kernel_size=3, stride=1,padding=1)

		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(2,2,0)


	def encoder(self,x, shortcut):
		
		debug("e0",x.shape)
		x =self.maxpool(self.relu(self.conv1(x)))

		debug("e1",x.shape)

		x =self.maxpool(self.relu(self.conv2(x)))
		if shortcut: return x

		debug(2,x.shape)
		x =self.maxpool(self.relu(self.conv3(x)))
		debug("latent=",x.shape)

		x =self.maxpool(self.relu(self.conv4(x)))
		x =self.maxpool(self.relu(self.conv5(x)))
		return x

	def decoder(self,x, shortcut):
		if not shortcut:
			x=self.relu(self.conv5up(self.deconv5(x)))
			x=self.relu(self.conv4up(self.deconv4(x)))
		
			debug(48,x.shape)
			x=self.relu(self.conv3up(self.deconv3(x)))
		
			debug(49,x.shape)
		
		x=self.relu(self.conv2up(self.deconv2(x)))
		
		debug(50,x.shape)
		
		x=self.relu(self.conv1up(self.deconv1(x)))
		debug(x.shape)
		return x
		

	def forward(self,x):
		# randomly only train the first and last few layers, skipping the middle
		# ths is not classic 'u-net' architecture, just simple encoder-decoder stack
		shortcut =  True #random.random()<0.5
		latent=self.encoder(x,shortcut)
		out=self.decoder(latent,shortcut)
		
		self.iter+=1
		if self.iter==self.nextshow:
			
			self.shown=transforms.ToPILImage()((x[0]).float()).show()
			self.shown=transforms.ToPILImage()((out[0]).float()).show()

			self.iter=0
			self.nextshow*=10

		return out
		

def check_ae_works():
	ae = Autoencoder()
	input = torch.randn(16,3,256,256)
	print("check : input=",input.shape,input.dtype)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)
	output = ae.forward(input)
	print("input=",input.shape, "\noutput=",output.shape)

class AddImageNoise(object):
	def __init__(self,amountfine,amountcoarse):
		self.amount=[amountfine,amountcoarse]
	def __call__(self,sample):
		rndlittle=((torch.rand( sample.shape)-0.5)*self.amount[0])
		resize=transforms.Resize((sample.shape[1],sample.shape[2]))
		rndbig=resize((torch.rand( sample.shape[0],int(sample.shape[1]/2),int(sample.shape[2]/2)) -0.5)*self.amount[1])
		rnd=rndlittle +rndbig

		sample=torch.add(sample,rnd)
		
		sample=torch.clip(sample,0,1.0)
		
		return sample
		

class NoisedImageDataset(Dataset):
	def __init__(self,dirname): 
		print("init dataset from dir: ",dirname)
		self.images=[]
		
		for i,x in enumerate(os.listdir(dirname)):
			img=Image.open(dirname+x)
			img=img.resize((255,255))
			imgarr=numpy.array(img)
			#imgarr=imgarr[0:255,0:255,0:3]
			imgarr=torch.tensor(imgarr.transpose((2,0,1))).float()*(1.0/255.0) # numpy HWC -> torch CHW
			if i<32:
				print("img[",i,"] ",x," size=",imgarr.shape)
			self.images.append(imgarr)
		print("total images=",len(self.images))

		self.add_noise=AddImageNoise(0.5,0.0)
			

	def __len__(self):
		return len(self.images)

	def __getitem__(self,idx):
		img=self.images[idx];
		return self.add_noise(img), img
		
		

def make_dataloader(dirname="../training_images/",show=False):

	dataset = NoisedImageDataset(dirname)
	if show:
		for x in range(0,4):
			(data,target)=dataset[x]
			transforms.ToPILImage()((data).float()).show()
			transforms.ToPILImage()((target).float()).show()
	return DataLoader(
		dataset,
		batch_size=4, 
		shuffle=True)
		
def  train_epoch(device, model,opt, dataloader):
	running_loss=0
	last_loss=0
	interval=10

       
	loss_function = torch.nn.MSELoss()

      # Storing the losses in a list for plotting

	for i,(data,target) in enumerate(dataloader):
		
		data.to(device)
		target.to(device)
#		for j in range(0,data.shape[0]):
		
		output=model(data)
		#loss=crit(output,target)
		loss = loss_function(output, target)
		opt.zero_grad()

		loss.backward()
		opt.step()

		running_loss+=loss.item()
		if i % interval == 0:
			print("i=",i, "\tloss=",running_loss/float(interval))
			running_loss=0
 
		



	
	

def main():
	print("initializing device..")

	no_mps=True
	if torch.cuda.is_available(): device=torch.device("cuda")
	elif torch.backends.mps.is_available() and not no_mps: device = torch.device("mps")
	else: device = torch.device("cpu")

	 
	print("using device:",device.type)

	print("grabbing dataset..")
	ae = Autoencoder()
	ae.to(device)
	
	dataloader=make_dataloader("../../training_images/")
	#optimizer = torch.optim.SGD(ae.parameters(), lr=0.01)

	optimizer = torch.optim.Adadelta(ae.parameters(), lr=0.1,weight_decay=1e-8)

	save_freq=5
	for i in range(0,10000):
		print("training epoch: ",i)
		train_epoch(device, ae,optimizer,  dataloader)
		if i%save_freq == 0:
			torch.save(ae.state_dict(), "trained/my_trained_ae_"+str((i/save_freq)%4))

	print("training on some images.")
	torch.save(ae.state_dict(), "my_trained_ae")

main()

		
