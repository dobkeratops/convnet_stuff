import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import psutil
import sys,getopt
import time
import matplotlib
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

import random
import os
from PIL import Image,ImageDraw,ImageFont
import socket

# so we can view progress on web plt.plot()
# launch on console, console can print handy link to view status.
def get_ip():
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.settimeout(0)
	try:
		# doesn't even have to be reachable
		s.connect(('10.254.254.254', 1))
		IP = s.getsockname()[0]
	except Exception:
		IP = '127.0.0.1'
	finally:
		s.close()
	return IP

def debug(*a):
	#print(*a)
	pass

def show_tensors_named(images_and_names):
	img = concat_named_images(images_and_names)
	img.show()

def to_pil_image(src):
        arr = src.detach().numpy()
        
        arr=numpy.transpose(arr.astype('uint8'),(1,2,0))
        print(arr.shape,arr.dtype)
        img = Image.fromarray(arr)
        return img

def concat_named_images(images_and_names):
	total_width=0
	max_height=0 
	
	
	for s,name in images_and_names:
		total_width+=s.shape[2]
		
		max_height=max(s.shape[1],max_height)

	dst_img=Image.new('RGB',(total_width,max_height))
	dx=0

	draw=ImageDraw.Draw(dst_img)
	for src_img,name in images_and_names:
		#tmp=to_pil_image((src_img).float())
		tmp=transforms.ToPILImage()((src_img).float())

		dst_img.paste(tmp, (int(dx),int((max_height+tmp.height)/2-tmp.height)))
		

		draw.text((dx,0),name, (0,255,0))
		dx+=tmp.width
	return dst_img

def add_title_to_image(title,img,bgcol=(192,192,192),fgcol=(64,64,64)):
	
	draw=ImageDraw.Draw(img)
	padding_y=16
	textbox= draw.getfont().getmask(title).getbbox()
	textsize=(textbox[2],textbox[3])

	
	width = max(img.width,textsize[0])
	dst=Image.new('RGB',(width,img.height+textsize[1]+padding_y), bgcol)
	draw=ImageDraw.Draw(dst)
	draw.text(((width-textsize[0])//2,padding_y/2),title, fgcol)
	dst.paste(img, ((width-img.width)//2, textsize[1]+padding_y) )
	return dst



class AutoEncoder(nn.Module):

	def eval_all(self,x):
		if False:
			self.iter+=1
			if self.iter==self.nextshow:
				self.nextshow+=128	
				
				show_tensors_named(
					[(x[0],"input+noise") if i==0 else (self.encode_decode(x,0,i)[0],"level"+str(i)) for i in range(0,self.levels+1)]
					
					)

			r = random.random()
			maxlevel =self.levels//2 if r<0.25 else self.levels-1 if (r<0.5 and self.levels>1) else self.levels
			out = self.encode_decode(x,0,maxlevel)
			return out

		else:
			self.iter+=1
			out=self.eval_unet(x)
			
			#if self.iter==self.nextshow:
			#	show_tensors_named([(x[0],"input + noise"),(out[0],"unet output["+str(self.levels)+"]")])
			#	self.nextshow+=128	

			return out
		

	
	def forward(self,x):
		return self.eval_all(x)

	def config_string(self):
		return "unet"+\
			" channels="+str(self.channels)+\
			" uselevels="+str(self.uselevel)+"/"+str(self.levels)+\
			" kernel="+str(self.kernelsize)+\
			" skip_connecitons="+str(self.skip_connections)+\
			" skip_dropout="+str(self.skip_dropout)

	def __init__(self,channels=[3,16,32,64,128],kernelSize= 5,skip_connections=False, skip_dropout=True):
		self.shown=None
		self.iter=0
		self.nextshow=5
		self.skip_dropout=skip_dropout
		self.channels=channels
		self.dropout=0.25
		
		self.kernelsize=kernelSize
		self.skip_connections=skip_connections
		
		super().__init__()
		imagec=3
		dim=32
		levels=len(channels)-1
		self.levels=levels
		self.uselevel=min(1,self.levels)
		
		#pytorch needs 'ModuleList' to find layers in arrays
		#self.downskip=nn.ModuleList([])
		self.conv = nn.ModuleList([nn.Conv2d(channels[i],channels[i+1], kernel_size=kernelSize, stride=1,padding='same')
				for i in range(0,levels)])

		# downskip is an attempt to provide skip connetions to &  from the latent space
		##for i in range(0,levels/2):
			
			
		
		
		self.downsample = nn.ModuleList(
			[nn.Conv2d(channels[i+1],channels[i+1], kernel_size=3, stride=2, padding=0)
				for i in range(0,levels)])

		self.upsample = nn.ModuleList(
			[nn.ConvTranspose2d(channels[i+1],channels[i+1], kernel_size=2,stride=2,padding=0,dilation=2)
				for i in range(0,self.levels)])

		maxchannels=channels[levels-1]
		
		self.convup = nn.ModuleList([nn.Conv2d(channels[i+1],channels[i], kernel_size=kernelSize, stride=1,padding='same')
				for i in range(0,self.levels)])
		
		print(self.conv, self.downsample, self.convup,self.upsample)
		self.maxpool = nn.MaxPool2d(2,2,0)
		self.avpool = nn.AvgPool2d(2,2,0)
		self.avpool3 = nn.AvgPool2d(3,2,0)
		self.activ = nn.ReLU()

	def encoder(self,x,inlevel,outlevel):
		x0=x
		debug("input shape=",x.shape)
		for i in range(inlevel,outlevel):
			if i>0: x=nn.Dropout(0.25)(x) #never dropout the actual input image channels!
			x=self.activ(self.conv[i](x))
			if i!=outlevel-1:
				x=self.downsample[i](x)
			debug("encoder[%d]shape output=" % (i),x.shape)

			
		return x

	def decoder(self,x,inlevel,outlevel):
		debug("latent shape=",x.shape)
		for i in reversed(range(inlevel,outlevel)):
			if self.dropout>0.01: x=nn.Dropout(self.dropout)(x)
			if i!=outlevel-1:
				if self.skip_connections:
					
					if not (self.skip_dropout and random.random()<0.5):
						x=torch.add(x,level_val[i])
						x*=0.5
				
				x=self.upsample[i](x)

			x=self.activ(self.convup[i](x))
			debug("decoder[%d]shape output=",x.shape)
		return x

	def encode_decode(self,x,inlevel,outlevel):
		latent  = self.encoder(x,inlevel,outlevel)
		return self.decoder(latent,inlevel,outlevel)

	def increase_depth(self):
		self.uselevel=min(self.uselevel+1,self.levels)
	
	def eval_unet(self, input):
		debug("eval unet ",input.shape)
		level_val=[]
		x=input
		
		
		for i in range(0,self.uselevel):
			debug("encode ",i)
			x=self.activ( self.conv[i](x))
			if i!=self.uselevel-1:
					x=self.downsample[i](x)
			debug("eval ubet", x.shape)
			level_val.append( x )
			

		x=level_val[self.uselevel-1]
		debug("latent", x.shape)
		for i in reversed(range(0,self.uselevel)):
			debug("decdode ",i)
			if i!=self.uselevel-1:
				if self.skip_connections:
					x=torch.add(x,level_val[i])
				x=self.upsample[i](x)
			if self.dropout>0.01: x=nn.Dropout(self.dropout)(x)
			x=self.activ(self.convup[i](x))

		return x
			
	def visualize_features(self):
		# make a 1-hot vector for each slot in inner most representation
		# convolute it with the expansion kernel 
		for i in range(0,levels):
			i


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

def make_noise_tensor(shape,sizediv=1, scale=1.0, deadzone=0.5,rgbness=0.5):
	#return (torch.rand(shape[0], int(shape[1]/sizediv),int(shape[2]/sizediv))*2.0-1.0)*scale  
	rgbness=random.random() #override, we will use both grey and rgb noise
	rgbn=False
	rnd=(torch.rand(1, int(shape[1]/sizediv),int(shape[2]/sizediv))*2.0-1.0)*(1-rgbness)
	rnd = torch.cat([rnd]*shape[0],0 )
	rnd+=(torch.rand(shape[0], int(shape[1]/sizediv),int(shape[2]/sizediv))*2.0-1.0)*rgbness

	rndAbove = torch.clip(rnd -deadzone,0.0,1.0)
	rndBelow = torch.clip(rnd +deadzone,-1.0,0.0)
	if deadzone<1.0:
		scale*=1.0/(1.0-deadzone)
	
	return (rndAbove+rndBelow)*scale

class AddImageNoise(object):
	def __init__(self,amounts_at_scales,noise_deadzone=0.5,rgbness=0.5):
		self.amounts=amounts_at_scales
		self.noise_deadzone=noise_deadzone
		self.rgbness=rgbness
		scale2x_weights = torch.ones([3,1,2,2])

		self.conv_scale2x = torch.nn.ConvTranspose2d(in_channels=3,out_channels=3,kernel_size=2,stride=2,groups=3,dilation=2)
		#self.conv_scale2x = torch.nn.Conv2d(3, 3, 2, 1, 0, 2, 1, False)
		self.conv_scale2x.weight = torch.nn.Parameter(scale2x_weights,False)

	def __call__(self,sample):
		# add noise at differnt scales, like fractal clouds
		# make it work harder to discover real patterns?
		resize=transforms.Resize((sample.shape[1],sample.shape[2]))

		noise_width=1
		for amount in self.amounts:
			rnd = make_noise_tensor(sample.shape, noise_width, amount,self.noise_deadzone,self.rgbness)

			rnd=resize(rnd)

			noise_width*=2
			sample = torch.add( sample, rnd )

		
		sample=torch.clip(sample,0,1.0)
		
		return sample
		

class NoisedImageDataset(Dataset):
	def __init__(self,dirname): 
		print("init dataset from dir: ",dirname)
		self.images=[]
		ishow=1
		for i,x in enumerate(os.listdir(dirname)):
			img=Image.open(dirname+x)
			img=img.resize((255,255))
			imgarr=numpy.array(img)
			#imgarr=imgarr[0:255,0:255,0:3]
			imgarr=torch.tensor(imgarr.transpose((2,0,1))).float()*(1.0/255.0) # numpy HWC -> torch CHW
			if i>=ishow:
				print("img[",i,"] ",x," size=",imgarr.shape)
				ishow*=2
			self.images.append(imgarr)
		print("total images=",len(self.images))

		self.add_noise=AddImageNoise([0.33,0.33,0.25,0.25,0.25],0.25,0.25)
			

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
			to_pil_image((data).float()).show()
			to_pil_image((target).float()).show()
	return DataLoader(
		dataset,
		batch_size=4, 
		shuffle=True)

def paste_vertical(src_imgs):
	maxw=0
	totaly=0
	for si in src_imgs:
		maxw=max(maxw,si.width)
		totaly+=si.height

	img=Image.new('RGB',(maxw,totaly),(128,128,128))
	dy=0
	for si in src_imgs:
		img.paste(si,((maxw-si.width)//2,dy))
		dy+=si.height
		
	return img

def visualize_progress(net,progress,time, loss, input_data, output, target):
	
	graph=progress.draw_graph()
	# if no display , store it in the local webserver ?
	img=add_title_to_image(
		"time="+("%.2f"%time)+"s loss="+("%.5f"%loss)+" "+net.config_string(),
			concat_named_images([(input_data," input + noise "),(output,"network output"), (target,"target")]),
		)
	img=paste_vertical([img,graph])
	if os.path.isdir("/var/www/html"):
		print("\tsee progress at http://",get_ip(),"/progress.jpg");
		img.save("/var/www/html/progress.jpg")

	img.show()

class Progress:
	def __init__(self):
		self.t_start = time.time()
		self.loss_graph=[]

	def time_elapsed(self):
		return time.time()-self.t_start
	
	def add_point(self,y):
		self.loss_graph.append((y,self.time_elapsed()))

	def draw_graph(self,size=(1024,256)):
		
		# plot with various axes scales
		plt.figure()

		# linear
		plt.subplot(221)
		
		plt.yscale('log')
		plt.ylabel('loss')
		plt.xlabel('time/s')
		plt.title('training progress')
		xs=[p[1] for p in self.loss_graph]
		ys=[p[0] for p in self.loss_graph]
		plt.plot(xs,ys)
		plt.grid(True)
		plt.savefig("loss_graph.jpg")
		plt.close()
		return Image.open("loss_graph.jpg")



def  train_epoch(device, model,opt, dataloader,progress):
	running_loss=0; num=0
	
	
	print_interval=min(10,len(dataloader)/2)
	
       
	loss_function = torch.nn.MSELoss()

      # Storing the losses in a list for plotting

	

	for i,(data,target) in enumerate(dataloader):
		
		data.to(device)
		target.to(device)
#		for j in range(0,data.shape[0]):
		
		output=model(data)
		#loss=crit(output,target)
		loss = loss_function(output, target)

		t_elapsed=progress.time_elapsed()
		
		#todo use timer..
		if model.iter==model.nextshow:
			visualize_progress(model,progress,t_elapsed,loss.item(),data[0],output[0],target[0])
			model.nextshow+=8
		
		opt.zero_grad()

		loss.backward()
		opt.step()

		running_loss+=loss.item(); num+=1

		if (i+1) % print_interval == 0:
			interval_av_loss= running_loss/float(num)
			progress.add_point(interval_av_loss)
			print("i=",i,"\tt=",t_elapsed,"\tloss=",interval_av_loss)
			if interval_av_loss<0.01:
				model.increase_depth()
				
			running_loss=0; num=0
			

	
def makedir(x): 
	if x[-1]!='/': x+='/'
	return x
	
def main(argv):

	inputdir,outputdir= "../../training_images_few/","trained_models/"
	learning_rate = 0.1
	try:
		opts, args = getopt.getopt(argv,"hi:o:r:",["indir=","outdir=","learningrate="])
	except getopt.GetoptError:
		print('test.py -i <inputfile> -o <outputfile> -l learningrate')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('test.py -i <inputdir> -o <outputdir>')
			sys.exit()
		elif opt in ("-i", "--indir"):
			inputdir = makedir(arg)
		elif opt in ("-i", "--indir"):
			
			outputdir = makedir(arg)
		elif opt in ("-l", "--outdir"):
			learningrate= arg

	print("initializing device..")


	no_mps=True
	if torch.cuda.is_available(): device=torch.device("cuda")
	elif torch.backends.mps.is_available() and not no_mps: device = torch.device("mps")
	else: device = torch.device("cpu")
	 
	print("using device:",device.type)

	print("building model:")
	ae = AutoEncoder(channels=[3,32,64,128],kernelSize= 5,skip_connections=True,skip_dropout=False)
	ae.uselevel=2
	print("model config:",ae.config_string())
	ae.to(device)


	print("grabbing dataset.."+inputdir)	
	dataloader=make_dataloader(inputdir)


	#optimizer = torch.optim.SGD(ae.parameters(), lr=0.01)

	optimizer = [torch.optim.Adadelta(ae.parameters(), lr=learning_rate,weight_decay=1e-8),
		torch.optim.Adadelta(ae.parameters(), lr=learning_rate*0.25,weight_decay=1e-8)]

	print("Start training LR:", learning_rate,"\tsaving to:\t"+outputdir)

	save_freq=5
	progress=Progress()
	for i in range(0,10000):
		print("training epoch: ",i)
		train_epoch(device, ae,optimizer[0 if i<200 else 1],  dataloader,progress)
		if (i+1) % 100==0: ae.increase_depth()
		if i%save_freq == 0:
			torch.save(ae.state_dict(), outputdir+"my_trained_ae_"+str((i/save_freq)%4))

	print("training on some images.")
	torch.save(ae.state_dict(), "my_trained_ae")

main(sys.argv[1:])

		
