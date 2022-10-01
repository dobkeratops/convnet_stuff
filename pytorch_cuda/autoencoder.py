from lib2to3.pgen2.tokenize import generate_tokens
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import psutil
import sys,getopt
import time
import matplotlib
#import torchvision
#from torchvision import transforms
from matplotlib import pyplot as plt

import random
import os
from PIL import Image,ImageDraw,ImageFont
import socket

g_show_interval=1024
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
        arr = src.cpu().detach().numpy()*255.0
        src.to(g_device)
        
        arr=numpy.transpose(arr.astype('uint8'),(1,2,0))
        
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
		tmp=to_pil_image((src_img).float())
		#tmp=transforms.ToPILImage()((src_img).float())

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

		

	
	def forward(self,x):
		self.iter+=1
		return self.eval_unet(x)
	


	def config_string(self):
		return	" channels="+str(self.channels)\
			+" ks="+str(self.kernelsize)+"x"+str(self.kernelsize)\
			+" use="+str(self.uselevel)+"/"+str(self.levels)\
			+" skipcon="+str(self.skip_connections)\
			+" dropout="+str(self.dropout)\

	def __init__(self,channels=[3,16,32,64,128],kernelSize= 5,skip_connections=False,dropout=0.25):
		self.shown=None
		self.iter=0
		self.nextshow=5
		self.channels=channels	# channels per layer
		self.dropout=dropout
		self.generate=False
		
		self.kernelsize=kernelSize
		self.skip_connections=skip_connections
		
		super().__init__()
		imagec=3
		dim=32
		levels=len(channels)-1
		self.levels=levels
		self.uselevel=max(1,self.levels)
		
		#pytorch needs 'ModuleList' to find layers in arrays
		#self.downskip=nn.ModuleList([])
		self.conv = nn.ModuleList([nn.Conv2d(channels[i],channels[i+1], kernel_size=kernelSize, stride=1,padding='same', device=g_device)
				for i in range(0,levels)])

		# downskip is an attempt to provide skip connetions to &  from the latent space
		##for i in range(0,levels/2):
			
			
		
		
		self.downsample = nn.ModuleList(
			[nn.Conv2d(channels[i+1],channels[i+1], kernel_size=3, stride=2, padding=0, device=g_device)
				for i in range(0,levels)])

		self.upsample = nn.ModuleList(
			[nn.ConvTranspose2d(channels[i+1],channels[i+1], kernel_size=2,stride=2, padding=0,dilation=2, device=g_device)
				for i in range(0,self.levels)])

		maxchannels=channels[levels-1]
		
		self.convup = nn.ModuleList([nn.Conv2d(channels[i+1],channels[i], kernel_size=kernelSize, stride=1,padding='same', device=g_device)
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
		
	def decrease_depth(self):
		self.uselevel=max(self.uselevel-1,1)
	
	def eval_unet(self, input):
		debug("eval unet ",input.shape)
		level_val=[]
		x=input
		debug("eval unet on input:where?",input.is_cuda)

		
		for i in range(0,self.uselevel):
			debug("encode ",i)
			x=self.conv[i](x)
			x=self.activ(x )
			if i!=self.uselevel-1:
					x=self.downsample[i](x)
			debug("eval ubet", x.shape)
			debug("size after encode[%d] ="%i,x.shape)
			level_val.append( x )
			

		x=level_val[self.uselevel-1]

		debug("latent", x.shape)
		if self.generate:
			x=torch.rand(x. shape)
			sparsity=0.1
			amplitude=100.0
			x=torch.sub(x,(1.0-sparsity))
			x=torch.mul(x,(1.0/(1.0-sparsity))*amplitude)
			x=torch.clip(x,0.0,amplitude)

		for i in reversed(range(0,self.uselevel)):
			debug("decdode ",i)
			if i!=self.uselevel-1:
				if self.skip_connections:
					x=torch.add(x,level_val[i])
				x=self.upsample[i](x)
			if self.dropout>0.01: x=nn.Dropout(self.dropout)(x)
			debug("size after decode [%d] ="%i,x.shape)
			x=self.activ(self.convup[i](x))
			
		# hack. todo - restore original encoder,decoder paths
		midx=level_val[2]
		for i in reversed(range(0,3)):
			debug("decdode ",i)
			if i!=self.uselevel-1:
				if self.skip_connections:
					midx=torch.add(midx,level_val[i])
				midx=self.upsample[i](midx)
			if self.dropout>0.01: midx=nn.Dropout(self.dropout)(midx)
			midx=self.activ(self.convup[i](midx))
		

		return (midx,x)
			
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
	(w,h)=(shape[1],shape[2]) if len(shape)==3 else (shape[2],shape[3])
	(w,h)=(w//sizediv,h//sizediv)
	debug("make rnd shape w,h=",w,h)
	rnd=(torch.rand(1, w,h, device=g_device)*2.0-1.0)*(1-rgbness)
	rnd.to(g_device)
	rnd = torch.cat([rnd]*shape[0],0 )
	rnd+=(torch.rand(shape[0], w,h,device=g_device)*2.0-1.0)*rgbness
	debug("make rnd shape ",rnd.shape)
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

		self.conv_scale2x = torch.nn.ConvTranspose2d(in_channels=3,out_channels=3,kernel_size=2,stride=2,padding=0,groups=3,dilation=2,device=g_device)
		#self.conv_scale2x = torch.nn.Conv2d(3, 3, 2, 1, 0, 2, 1, False)
		ssize= self.conv_scale2x.weight.shape
		
		

		scale2x_weights = torch.ones(ssize,device=g_device)
		self.conv_scale2x.weight = torch.nn.Parameter(scale2x_weights,False)

	def __call__(self,sample):


		# add noise at differnt scales, like fractal clouds
		# make it work harder to discover real patterns?
		#resize=transforms.Resize((sample.shape[1],sample.shape[2]))

		noise_width=1
		#todo - extract o fractal noise function
		for n,amount in enumerate(self.amounts):
			debug("make noise tensor for ",sample.shape)
			rnd = make_noise_tensor(sample.shape, noise_width, amount,self.noise_deadzone,self.rgbness)
			debug("rnd shape level[%d]" % n, rnd.shape)

			#rnd=resize(rnd)
			if False:
				for i in range(0,n):
					shape=rnd.shape
					#print("shapes:",rnd.shape, sample.shape,shape)
					rnd=torch.reshape(rnd,(1,shape[0],shape[1],shape[2]))
					#print("scaling:" ,self.conv_scale2x.weight.shape,rnd.shape)
					rnd=self.conv_scale2x(rnd)
					#print("scaling:->",rnd.shape)
				
			rnd.to(g_device)
			if noise_width>1:
				debug("rnd shape", rnd.shape)
				rnd=torch.reshape(rnd,(1,rnd.shape[0],rnd.shape[1],rnd.shape[2]))
				debug("rnd shape pre resize", rnd.shape)
				#print("inshape", rnd.shape,sample.shape)
				debug ("sample shape pre resize:",sample.shape)
				wh = (sample.shape[1],sample.shape[2]) if len(sample.shape) ==3 else (sample.shape[2],sample.shape[3])
				debug ("rndshape:",rnd.shape)
				rnd = torch.nn.functional.interpolate(rnd, wh,mode='bilinear', align_corners=False)
			noise_width*=2

			#print("outshape", sample.shape, rnd.shape)
			sample = torch.add( sample, rnd);#torch.reshape(rnd,(sample.shape[1],sample.shape[2],sample.shape[3])) )
			

		
		sample=torch.clip(sample,0,1.0)
		if len(sample.shape)==4:
			sample = torch.reshape(sample, (sample.shape[1],sample.shape[2],sample.shape[3]))
		
		
		return sample
		

class NoisedImageDataset(Dataset):
	def __init__(self,dirname,max=1000000000): 
		print("init dataset from dir: ",dirname)
		self.images=[]
		ishow=1
		for i,x in enumerate(os.listdir(dirname)):
			if i>max: break
			img=Image.open(dirname+x)
			img=img.resize((255,255))
			imgarr=numpy.array(img)
			#imgarr=imgarr[0:255,0:255,0:3]
			imgarr=torch.tensor(imgarr.transpose((2,0,1)),device=g_device).float()*(1.0/255.0) # numpy HWC -> torch CHW

			if i>=ishow:
				print("img[",i,"] ",x," size=",imgarr.shape,"device:",imgarr.device)
				ishow*=2
			self.images.append(imgarr)
		print("total images=",len(self.images))

		self.add_noise=AddImageNoise([0.33,0.33,0.25,0.25,0.25],0.25,0.25)
			

	def __len__(self):
		return len(self.images)

	def __getitem__(self,idx):
		img=self.images[idx];
		return self.add_noise(img), img
		
		

def make_dataloader(dirname="../training_images/",show=False,max=1000000000):

	dataset = NoisedImageDataset(dirname,max)
	if show:
		for x in range(0,4):
			(data,target)=dataset[x]
			to_pil_image((data).float()).show()
			to_pil_image((target).float()).show()
	return DataLoader(
		dataset,
		batch_size=4, 
		shuffle=True)

def paste_images_vertical(src_imgs):
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


def make_progress_page(net,progress):	#this is just a page to make it refresh the image.
	# TODO actually we could just rely on html layout
	# instead of manually pasting the image
	return "	\
<!DOCTYPE html>\
<html>\
  \
<head>\
    <title>Page Title</title>\
    <meta http-equiv='refresh' content='4'>\
</head>\
  \
<body>\
	<img src='training_progress.jpg'></img>\
</body>\
</html>\
"

def visualize_progress(net,progress,time, loss, input_data, output, target):
	
	graph=progress.draw_graph()
	# if no display , store it in the local webserver ?
	img=add_title_to_image(
		"time="+("%.2f"%time)+"s loss="+("%.5f"%loss)+" "+net.config_string(),	
		paste_images_vertical([
			concat_named_images([(input_data," input + noise "),(output[0],"network output Mid"),(output[1],"network output Final"), (target,"target")]),
			graph]))
	img.save("training_progress.jpg")	# save the progress image in the current working director regardless. (TODO,save alongside net.)
	if os.path.isdir("/var/www/html"):
		img.save("/var/www/html/training_progress.jpg")
		print("\tsee progress at http://"+str(get_ip())+"/training_progress.html");
		f=open("/var/www/html/training_progress.html","w")
		f.write(make_progress_page(net,progress))
		f.close()
	else:
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
		
		plt.subplot(221)
		plt.yscale('log')
		plt.ylabel('loss')
		plt.xlabel('time/s')
		halfway=len(self.loss_graph)//2
		xs=[p[1] for p in self.loss_graph[halfway:]]
		ys=[p[0] for p in self.loss_graph[halfway:]]
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
		debug("where is data,target?",device, data.is_cuda,target.is_cuda)
#		for j in range(0,data.shape[0]):
		
		#print(data.get_device(),target.get_device())
		#print( model.conv[0].get_device())
		output=model(data)
		#loss=crit(output,target)

		loss = loss_function(output[0], target)+loss_function(output[1],target)

		t_elapsed=progress.time_elapsed()
		
		#todo use timer..
		if model.iter==model.nextshow:
			visualize_progress(model,progress,t_elapsed,loss.item(),data[0],(output[0][0],output[1][0]),target[0])
			model.nextshow+=g_show_interval

			
		
		opt.zero_grad()

		loss.backward()
		opt.step()

		running_loss+=loss.item(); num+=1

		if (i+1) % print_interval == 0:
			interval_av_loss= running_loss/float(num)
			progress.add_point(interval_av_loss)
			print("i=",i,"\tt=",t_elapsed,"\tloss=",interval_av_loss)
#			if interval_av_loss<0.007:
#				model.increase_depth()
#			if interval_av_loss>0.02:
#				model.decrease_depth()

				
			running_loss=0; num=0

def load_model(model,filename):
	print("loading pretrained model: ",filename)
	loaded_statedict=torch.load(filename,map_location=g_device)
	print("\nmodel has items:")
	
	not_found=[]
	not_matched=[]
	for key in loaded_statedict:
		if key in model.state_dict():
			ls=loaded_statedict[key].shape
			ms=model.state_dict()[key].data.shape
			if ls !=ms:
				not_matched.append(key)
			else:
				model.state_dict()[key].data = loaded_statedict[key].data

				print("\t",key,"\t", ms,"\t-ok",)
		else:
			not_found.append(key)
	
	for key in model.state_dict():
		if not key in loaded_statedict:
			print("(model key\t",key,"\tnot in file)")
		
	if len(not_found)>0:
		print("current model does not contain pretrained file nodes:",not_found)
		exit(0)

	if len(not_matched)>0:
		print("current model shapes do not match those in file:",not_found)
		for key in not_matched:
			ls=loaded_statedict[key].shape
			ms=model.state_dict()[key].data.shape

			print("\t",key,"",ls,"vs",ms)

		exit(0)
	model.load_state_dict(loaded_statedict)
	print("file",filename,"loaded fine..")

	
def makedir(x): 
	if x[-1]!='/': x+='/'
	return x
	
def main(argv):
	
	

	inputdir,outputdir,pretrained= "../training_images/","current_model/",None
	learning_rate = 0.1
	try:
		opts, args = getopt.getopt(argv,"hi:o:r:p:",["indir=","outdir=","learningrate=","pretrained"])
	except getopt.GetoptError:
		print('useage: test.py -i <inputdir> -o <outputdir> -l learningrate -p <pretrained>')
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
		elif opt in ("-p", "--pretrained"):
			print("setting pretrainde model =",arg)
			pretrained = arg

	print("initializing device..")


	no_mps=True
	if torch.cuda.is_available(): device=torch.device("cuda")
	elif torch.backends.mps.is_available() and not no_mps: device = torch.device("mps")
	else: device = torch.device("cpu")
	 
	print("using device:",device.type)
	global g_device
	g_device=device
	print("building model:")
#	ae = AutoEncoder(channels=[3,32,64,128,256],kernelSize= 7,skip_connections=True,skip_dropout=False)
#	ae = AutoEncoder(channels=[3,32,64,128,256],kernelSize= 5,skip_connections=False)
	ae = AutoEncoder(channels=[3,16,32,64,128,128],kernelSize= 5,skip_connections=False)

	
	if pretrained !=None:
		print("loading pretrained model:",pretrained)
		load_model(ae,pretrained)

	print("model config:",ae.config_string())
	ae.to(device)


	for i,c in enumerate(ae.conv): print("layer[%d] is cuda?"%i,c.weight.is_cuda)


	print("grabbing dataset.."+inputdir)	
	dataloader=make_dataloader(inputdir)


	#optimizer = torch.optim.SGD(ae.parameters(), lr=0.01)

	optimizer = [torch.optim.Adadelta(ae.parameters(), lr=learning_rate,weight_decay=1e-8),
		torch.optim.Adadelta(ae.parameters(), lr=learning_rate*0.25,weight_decay=1e-8)]

	print("Start training LR:", learning_rate,"\tsaving to:\t"+outputdir)

	save_freq=40 #todo save frequency based on time, once every 10 seconds raeehr than iterations..
	progress=Progress()
	for i in range(0,50000):
		print("training epoch: ",i)
		train_epoch(device, ae,optimizer[0],  dataloader,progress)
		
		if (i+1) % 100==0: ae.increase_depth()
		if i%save_freq == 0:
			torch.save(ae.state_dict(), outputdir+"my_trained_ae_"+str((i/save_freq)%8))

	print("training on some images.")
	torch.save(ae.state_dict(), "my_trained_ae")

main(sys.argv[1:])

		
