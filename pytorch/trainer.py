import random

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
import os
from PIL import Image,ImageDraw,ImageFont
import socket
import webbrowser

g_page_opened=False
g_show_interval=10
g_show_plot=0
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
	img = concat_named_images_horiz(images_and_names)
	img.show()



def add_color_sat(s0,s1):
	r0,g0,b0=s0
	r1,g1,b1=s1
	clamp255 = lambda a: max(0,min(a,255))
	return (clamp255(r0+r1),clamp255(g0+g1),clamp255(b0+b1))

def to_pil_image(src,bgcol=(128,128,128),cascaded=True,border=8):
	arr = src.cpu().detach().numpy()*255.0
	src.to(g_device)

	
	arr=numpy.transpose(arr.astype('uint8'),(1,2,0))

	num_groups=(arr.shape[2]+2)//3;
	subbg=add_color_sat(bgcol,(-16,-16,-16))
	if cascaded:
		z=max(arr.shape[0]//2,arr.shape[1]//2)
		z=min( z,max(4,max(arr.shape[0],arr.shape[1])//(num_groups)) )
		iw,ih=arr.shape[0],arr.shape[1]
		w=(num_groups-1)*z+iw+border*2
		h=(num_groups-1)*z+ih+border*2
		img=Image.new('RGB',(w,h),bgcol)
		maxk = arr.shape[2]-1
		#img.paste(subbg,[1,1,w-1,h-1])
		draw=ImageDraw.Draw(img)
		sb=border/2
		draw.polygon([(sb,sb),(iw+sb,sb),(w-sb,h-sb-ih),(w-sb,h-sb),(w-sb-iw,h-sb),(sb,ih+sb)], subbg)
		for i in range(0,num_groups):
		
			#np.take(arr, indices, axis=3) is equivalent to arr[:,:,:,indices,...].
			
			subimg = Image.fromarray(numpy.take(arr, [min(i*3+0,maxk),min(i*3+1,maxk),min(i*3+2,maxk)], 2))
			img.paste(subimg,(i*z+border,i*z+border))

		#img = Image.fromarray(arr)
		return img
	else:
		print("todo non cascaded vis in to_pil_image")
		exit(0)

def concat_named_images_horiz(images_and_names):
	total_width=0
	max_height=0 
	
	bgcol=(128,128,128)
	
	images=[(name,to_pil_image(a.float(),bgcol)) for name,a in images_and_names]
	
	for name,s in images:
		total_width+=s.width
		max_height=max(s.height,max_height)
	
	dst_img=Image.new('RGB',(total_width,max_height),bgcol)
	dx=0

	
	draw=ImageDraw.Draw(dst_img)
	for name,src_img in images:
		#tmp=transforms.ToPILImage()((src_img).float())
		dst_img.paste(src_img, (int(dx),int((max_height+src_img.height)/2-src_img.height)))

		draw.text((dx,0),name, (0,255,0))
		dx+=src_img.width

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

class Encoder(nn.Module):
	def __init__(self,channels=[3,16,32,64,128],kernelSize=5,dropout=0.25,downsample_kernel_size=3):
		super().__init__()
		self.channels=channels
		self.kernel_size=kernelSize
		self.dropout=dropout
		self.downsample_kernel_size=downsample_kernel_size
		self.levels = len(channels)-1

		print("make Encoder conv layers")
		self.conv = nn.ModuleList([nn.Conv2d(channels[i],channels[i+1], kernel_size=kernelSize, stride=1,padding='same', device=g_device)
				for i in range(0,self.levels)])
		print("make encoder downsamplers")

		if self.downsample_kernel_size>0:
			ds=[]
			for i in range(0,self.levels):
				ds.append(nn.Conv2d(channels[i+1],channels[i+1], kernel_size=self.downsample_kernel_size, stride=2, padding=0, device=g_device))
			self.downsample = nn.ModuleList(ds)
		else:
			self.downsample = None		
		self.maxpool = nn.MaxPool2d(2,2,0)
		self.activ = nn.ReLU()

	def forward(self,x):
		level_val=[]
		for i in range(0,self.levels):
			debug("encode ",i)
			x=self.conv[i](x)
			x=self.activ(x )
			#if i!=self.uselevel-1:
				
			x=self.downsample[i](x) if self.downsample else self.maxpool(x)
			debug("eval ubet", x.shape)
			debug("size after encode[%d] ="%i,x.shape)
			level_val.append( x )

#		x=level_val[self.uselevel-1]
		return level_val

class Decoder(nn.Module):
	def __init__(self,channels=[3,16,32,64,128],kernelSize=5,dropout=0.25,downsample_kernel_size=3,skip_connections=True):
		super().__init__()
		self.skip_connections=skip_connections
		self.channels=channels
		self.kernel_size=kernelSize
		self.dropout=dropout
		self.downsample_kernel_size=downsample_kernel_size
		self.levels = len(channels)-1

		self.maxpool = nn.MaxPool2d(2,2,0)
		self.activ = nn.ReLU()
		self.upsample = nn.ModuleList(
			[nn.ConvTranspose2d(channels[i+1],channels[i+1], kernel_size=2,stride=2, padding=0,dilation=2, device=g_device)
				for i in range(0,self.levels)])

		self.conv = nn.ModuleList([nn.Conv2d(channels[i+1],channels[i], kernel_size=kernelSize, stride=1,padding='same', device=g_device)
				for i in range(0,self.levels)])


	def forward(self,level_val):

		x=level_val[len(level_val)-1]
		for i in reversed(range(0,self.levels)):
			if i!=self.levels-1:
				if self.skip_connections:
					x=self.activ(x+level_val[i])
			x=self.upsample[i](x)
			if self.dropout>0.01: x=nn.Dropout(self.dropout)(x)
			x=self.activ(self.conv[i](x))

		return x

def count_params(net):
	num=0
	for p in list(net.parameters()):
		nn=1
		for s in list(p.size()):
			nn = nn*s
		num += nn
	return num

class UnetLatentTransform:
	def forward(self,level_vals):

		output_level_vals=[]
		for i in range(self.levels):
			output_level_vals.append(self.conv[i](level_vals[i]))

		return output_level_vals

	def __init__(self,encoder,decoder,kernel_size=1,inner_kernel_size=3):
		super().__init__()
		assert encoder.levels == decoder.levels
		self.levels=encoder.levels
		self.conv = nn.ModuleList(
			[	
				nn.Conv2d(encoder.channels[i+1],decoder.channels[i+1], kernel_size=kernel_size if i!=self.levels-1 else inner_kernel_size, stride=1, padding='same', device=g_device)
				for i in range(0,self.levels)
			]
		)


class EncoderDecoder(nn.Module):
	def forward(self,input):
		self.iter+=1

		latent_vals=self.encoder.forward(input)
		transformed_latent_vals= self.main_task.forward(latent_vals)
		x=self.decoder.forward(transformed_latent_vals)

		return [('final',x)]
		

	def estimate_cost(self): #argh now i find myself apreciating rust's insistance on param structs etc.
		total=0
		quotient=1
		params=0
		ksize=self.kernelsize
		encoder_ch,decoder_ch = self.encoder.channels,self.decoder.channels
		dsksize=self.downsample_kernel_size
		
		for i in range(0,len(encoder_ch)-1):
			
			x=encoder_ch[i]*(1+encoder_ch[i+1])*ksize*ksize 
			x+=encoder_ch[i+1]*encoder_ch[i+1]*dsksize*dsksize	#downsample
			params+=x
			total+=x/quotient
			quotient*=2*2

		quotient=1
		for i in range(0,len(decoder_ch)-1):
			x=decoder_ch[i]*(1+decoder_ch[i+1])*ksize*ksize
			x+=decoder_ch[i+1]*decoder_ch[i]*4 #upsample
			if i>0: #skip connection combiner cost
				x+=encoder_ch[i]*decoder_ch[i]
			params+=x
			total+=x/quotient
			quotient*=2*2
	
		return {'flops_per_output_pixel':total,'params':params,'reported_params':count_params(self)}
		

	def config_string(self):
		return	" channels=("+str(self.encoder.channels)+","+str(self.decoder.channels)+")"\
			+" ks="+str(self.kernelsize)+"x"+str(self.kernelsize)\
			+" use="+str(self.uselevel)+"/"+str(self.levels)\
			+" skipcon="+str(self.skip_connections)\
			+" dropout="+str(self.dropout)\
			+" dsks="+str(self.downsample_kernel_size)

	def __init__(self,encoder_ch=[3,16,32,64,128],decoder_ch=[128,64,32,16,3],kernelSize= 5,skip_connections=False,dropout=0.25,downsample_kernel_size=3):
		super().__init__()
		#todo - this config stuff into a shared object?
		#we copy a lot of this into the encoder/decoder parts.
		self.shown=None
		self.iter=0
		self.nextshow=1
		self.dropout=dropout
		self.downsample_kernel_size=downsample_kernel_size
		
		self.kernelsize=kernelSize
		self.skip_connections=skip_connections

		latent_dim=encoder_ch[len(encoder_ch)-1]
		#self.latent_space_vis = nn.ModuleList([nn.Conv2d(latent_dim,3,1),nn.Conv2d(3,latent+dim,1)])
		#TODO an internal translator layer between 'encoder_ch' , 'decoder_ch'
		# such that they dont actually have to be the same dimensionality
		# they will eventually be task branches.
		
		imagec=3
		dim=32
		levels=len(encoder_ch)-1
		assert(len(encoder_ch)==len(decoder_ch))
		self.levels=levels
		self.uselevel=max(1,self.levels)

		#pytorch needs 'ModuleList' to find layers in arrays
		#self.downskip=nn.ModuleList([])
		
		print("make encoder ",encoder_ch)
		self.encoder = Encoder(encoder_ch,kernelSize,dropout,downsample_kernel_size)
		print("make decoder ",decoder_ch)
		self.decoder = Decoder(decoder_ch,kernelSize,dropout)

		print("make latent space transform (main task)")
		self.main_task = UnetLatentTransform(self.encoder,self.decoder, kernel_size=1)
		
		cost=self.estimate_cost() 
		print("est cost=",cost)
		print("create EncoderDecoder done")

def dump_state_dict(net):
	print("state_dict():")
	for k in net.state_dict():
		v=net.state_dict()[k]
		print("node:",k,v.size())


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


def load_image_as_tensor(i, dirname,fname,size):
	_fname=dirname+"/"+fname;
	
	if not os.path.exists(_fname):
		for x in [".jpg",".JPG",".jpeg",".JPEG",".png",".PNG"]:
			if os.path.exists(_fname+x):
				_fname+=x
				break
		else:
			print("error could not load:",_fname)
			return None
	
	img=Image.open(_fname)
	img=img.resize((size,size))
	imgarr=numpy.array(img)
	#imgarr=imgarr[0:255,0:255,0:3]
	imgarr=torch.tensor(imgarr.transpose((2,0,1)),device=g_device).float()*(1.0/255.0) # numpy HWC -> torch CHW

	if i%32==0:
		print("img[",i,"] ",fname," size=",imgarr.shape,"device:",imgarr.device)

	return imgarr

def name_ext(fname):
	x=fname.rfind(".")
	return fname[0:x],fname[x+1:]

class TransformImageDataset(Dataset):
	def init_simple(self):

		find_image_pairs={}
		for i,fname in enumerate(os.listdir(dirname)):
			if fname[0]=='.':
				continue

			if not "_OUTPUT" in fname:
				fname0,ext=name_ext(fname)
				find_image_pairs[fname]=fname0+"_OUTPUT"+"."+ext


		for k in find_image_pairs:
			print(k, " -> ",find_image_pairs[k])
			img_in=load_image_as_tensor(i,dirname,k,255)
			img_out=load_image_as_tensor(i,dirname,find_image_pairs[k],255)
			assert img_in.shape[0]==img_out.shape[0] and img_in.shape[1]==img_out.shape[1],"different sizes for input & output not supported yet(WIP)"
			
			if img_out is None:
				print("warning no _OUTPUT for ",k)
			else:
				self.image_pairs.append((img_in,img_out))

		

	def lookup_basename_postfix(self,basenames,name):
		for i in reversed(range(0,len(name))):
			name[:name.rfind('.')]
			if name[i]=='_':
				if name[0:i] in basenames:
					return name[0:i],name[i:]
		
		return None,None

	def io_channels(self):
		if len(self.image_io_pairs)==0: return (0,0)
		first_in,first_out=self.image_io_pairs[0]
		return (first_in.shape[0],first_out.shape[0])

	def __init__(self,dirname,max_files=10000000,io_postfixes=None):
		#TODO option to initialise from directory tree eg dirname/input/*  dirname/output/* , or dirname/foo/input.jpg,output.jpg dirname/bar/jpg etc
		if io_postfixes:
			self.init_from_filenames(dirname,max_files,io_postfixes)
		else:
			self.init_from_filenames(dirname,max_files)

	def init_from_filenames(self,dirname,max_files=10000000,
			input_output_postfixes=(
				#generic names +ideas for what we'll try to generate to train on..
				["_INPUT0","_INPUT1","_INPUT2","_INPUT3","_INPUT",
				"_IN_NORMAL","_IN_VEL","_IN_PBR","_IN_AO","_IN_DEPTH","_IN_SHADOW", 
				"_IN_LIGHT_DX","_IN_LIGHT_DY","_IN_LIGHT_DZ","_IN_LIGHT"],
				["_OUTPUT0","_OUTPUT1","_OUTPUT2","_OUTPUT3","_OUTPUT",
				"_OUT_DEPTH","_OUT_VEL","_OUT_NORMAL","_OUT_PBR","_OUT_AO","_OUT_SHADOW", 
				"_OUT_LIGHT_DX","_OUT_LIGHT_DY","_OUT_LIGHT_DZ","_OUT_LIGHT"]),scale_to=255):
		print("init dataset from dir: using filename extensions ",dirname)
		self.image_io_pairs=[]
		#self.init_simple()
		
		basenames={}
		channelnames={}
		potential_input_postfixes,potential_output_postfixes=input_output_postfixes

		print("finding output images..")

		# todo support either input or output driven
		# eg foo_[many input names] -> foo_OUTPUT
		# or foo_INPUT -> foo_[many output names]

		used_inputs={}	# track which input/output postfixes we have
		used_outputs={}

		def insert_or_inc(ks,k):
			if k in ks: ks[k]+=1
			else: ks[k]=1

		for i,fname in enumerate(os.listdir(dirname)):
			if fname[0]=='.': continue
			
			for n,pfx in enumerate(potential_input_postfixes):
				if pfx in fname:
					basename=fname[:fname.rfind(pfx)]
					insert_or_inc(basenames,basename)
					used_inputs[n]=True
					break

			for n,pfx in enumerate(potential_output_postfixes):
				if pfx in fname:
					basename=fname[:fname.rfind(pfx)]
					insert_or_inc(basenames,basename)
					used_outputs[n]=True
					break

		input_postfixes=[]
		output_postfixes=[]
		for n,name in enumerate(potential_input_postfixes):
			if n in used_inputs: input_postfixes.append(name)

		for n,name in enumerate(potential_output_postfixes):
			if n in used_outputs: output_postfixes.append(name)
		
		if len(input_postfixes)==0: input_postfixes.append("")
		elif len(output_postfixes)==0: output_postfixes.append("")

		# for each basename: read the inputs..
		print("used channels:",input_postfixes,output_postfixes)
		print(basenames)
		
		used_chans=0
		for x in basenames:
			used_chans=max(used_chans,basenames[x])
		fail=False
		for x in basenames:
			if basenames[x]!=used_chans:
				print("error basename:\t",x,"\thas missing channels\n")
				fail=True
		if fail: 
			print("all images must have the same input/output channels supplied")
			exit(0)

		for f in basenames:
			inputs=[]
			outputs=[]
			for pfx in input_postfixes:
				
				inputs.append(load_image_as_tensor(i,dirname,f+pfx,scale_to))

			for pfx in output_postfixes:
				x=load_image_as_tensor(i,dirname,f+pfx,scale_to)
				outputs.append(x)

			input = torch.cat(inputs,0)
			output= torch.cat(outputs,0)
			print(basename,"\t:io shapes=", input.shape,output.shape)
			#to_pil_image(output).show()
			
			self.image_io_pairs.append((input,output))



	def __len__(self):
		return len(self.image_io_pairs)

	def __getitem__(self,idx):
		pair=self.image_io_pairs[idx]
		return pair

class NoisedImageDataset(Dataset):
	def __init__(self,dirname,max=1000000000,noise=0.33): 

		print("init dataset from dir: ",dirname)
		self.images=[]
		
		for i,fname in enumerate(os.listdir(dirname)):
			if i>max: break
			if fname[0]=='.':
				continue
			img=load_image_as_tensor(i,dirname,fname,255)
			self.images.append(img)
		print("total images=",len(self.images))

		
		#self.add_noise=AddImageNoise([0.33,0.33,0.25,0.25,0.25],0.25,0.25)
		self.add_noise=AddImageNoise([noise],0.25,0.25)
		

	def __len__(self):
		return len(self.images)

	def __getitem__(self,idx):
		img=self.images[idx];
		return self.add_noise(img), img

def make_dataloader(dirname="../training_images/",show=False,max=1000000000,noise=0.33):

	is_transform_pairs=	False
	print("dataset: ",dirname,"\tlen=",len(os.listdir(dirname)))
	for i,fname in enumerate(os.listdir(dirname)):
		if "_OUTPUT" in fname or "_INPUT" in fname:
			is_transform_pairs=True

	dataset=None
	if is_transform_pairs:
		print("input/_OUTPUT pairs detected - setting up dataloader for image transformation")
		dataset=TransformImageDataset(dirname)
	else:
		print("plain image directory - creating noised image dataloader for autoencoder training")
		n=noise
		dataset = NoisedImageDataset(dirname,max,noise=n)

	if show:
		for x in range(0,4):
			(data,target)=dataset[x]
			to_pil_image((data).float()).show()
			to_pil_image((target).float()).show()

	return DataLoader(
		dataset,
		batch_size=min(4,len(dataset)), 
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
def from_batch(pfx_name,ab,index):
	return (pfx_name+":"+ab[0], ab[1][index])

def color_to_float(rgb): 
	r,g,b=rgb
	return (float(r)*(1.0/255.0),float(g)*(1.0/255.0),float(b)*(1.0/255.0))



def visualize_progress(net,progress,time, loss, input_data, output, target):
	
	
	# if no display , store it in the local webserver ?
	
	images=concat_named_images_horiz([("input",input_data)]+[from_batch("output:",out,0) for out in output ]+[ ("target",target)])
	#,from_batch("output",output[1],0), ("target",target)])

	graph=progress.draw_graph(size=(images.width,images.width/3))
	img=add_title_to_image(
		"time="+("%.2f"%time)+"s loss="+("%.5f"%loss)+" "+net.config_string(),	
		paste_images_vertical([
			images,
			graph]))
	img.save("training_progress.jpg")	# save the progress image in the current working director regardless. (TODO,save alongside net.)
	img.save(progress.outputdir+"training_progress.jpg")

	serving=True
	progressdir="var/www/html"
	if not os.path.isdir(progressdir):
		progressdir = os.getcwd()
		serving=False
	else:
		print("\tsee progress at http://"+str(get_ip())+"/training_progress.html");
	if progressdir[-1]!="/": progressdir+="/"

	img.save(progressdir+"training_progress.jpg")
	img.save(progress.outputdir+"training_progress.jpg")	# save it alongside the output aswell
	
	f=open(progressdir+"training_progress.html","w")
	f.write(make_progress_page(net,progress))
	f.close()
	global g_page_opened
	#todo - check if headless or what.
	if not g_page_opened:
		g_page_opened=True
		webbrowser.open("file:///"+progressdir+"training_progress.html")
	return;


class Progress:
	def __init__(self,saveto="./"):
		self.t_start = time.time()
		self.loss_graph=[]
		self.outputdir=saveto

	def time_elapsed(self):
		return time.time()-self.t_start
	
	def add_point(self,y):
		self.loss_graph.append((y,self.time_elapsed()))

	def draw_graph(self,size=(1024,256),bgcol=(128,128,128)):
		
		# plot with various axes scales
		fig=plt.figure()
		fig.set_facecolor(color_to_float(bgcol))
		dpi=100 #wtf
		fig.set_size_inches(size[0]/dpi,size[1]/dpi)

		
		#plt.subplot((1,1,1))
		
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

		#myaximage = ax.imshow(im,
         #             aspect='auto',
                      #extent=(20, 80, 20, 80),
                      #alpha=0.5)

		return Image.open("loss_graph.jpg")



def  train_epoch(device, model,opt, dataloader,progress):
	running_loss=0; num=0
	
	print_interval=min(10,len(dataloader)/2)
	
       
	loss_function = torch.nn.MSELoss()

      # Storing the losses in a list for plotting

	

	for i,(data,target) in enumerate(dataloader):
		
		data=data.to(device)
		target=target.to(device)
		debug("where is data,target?",device, data.is_cuda,target.is_cuda)
#		for j in range(0,data.shape[0]):
		
		#print(data.get_device(),target.get_device())
		#print( model.conv[0].get_device())
		output=model(data)
		#_,shortcut_out=output[0]
		#_,final_out=output[1]
		#loss=crit(output,target)

		# todo - multitask trainng - some targets will be missing..

		# handle loss from a training process that might return a shortcut
		# TODO , those key-value pairs will eventually denote different TASKS, which may or may not be present
		loss =0.0
		for name,outval in output:
			loss += loss_function(outval, target)# loss_function(shortcut_out, target)+loss_function(final_out,target)

		t_elapsed=progress.time_elapsed()
		
		#todo use timer..
		if model.iter==model.nextshow:
			visualize_progress(model, progress, t_elapsed, loss.item(), data[0], output, target[0])
			model.nextshow+=g_show_interval
		
		opt.zero_grad()

		loss.backward()
		opt.step()

		running_loss+=loss.item(); num+=1

		if (i+1) % print_interval == 0:
			interval_av_loss= running_loss/float(num)
			progress.add_point(interval_av_loss)
			print("i=",i,"\tt=",t_elapsed,"\tloss=",interval_av_loss)
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

def foo():
	print("foo")	#test something

def make_layer_channel_list(io_channels,_input_features,_latent_depth,layers):
	# these can be different because input,output can have different dimensionality and resolution
	inch,outch=io_channels
	
	ld=_input_features
	if _latent_depth==0: _latent_depth=_input_features*(2**layers)
	
	encoder_ch=[inch]
	for i in range(0,layers-1):
		encoder_ch.append(min(ld,_latent_depth))
		ld*=2
	encoder_ch.append(_latent_depth)

	#todo count this the other way, so the ramp works with different depth
	# currently its duplicate of above logic :/
	ld = _latent_depth
	decoder_ch=[outch]
	ld=_input_features
	for i in range(0,layers-1):
		decoder_ch.append(min(ld,_latent_depth))
		ld*=2
	decoder_ch.append(_latent_depth)
	print(encoder_ch,decoder_ch)

	return (encoder_ch,decoder_ch)



def main(argv):

	inputdir,outputdir,pretrained= "../training_images/","current_model/",None
	learning_rate = 0.1
	noise_amplitude=0.33
	_dropout=0.25
	_latent_depth=256
	_input_features=16
	_kernel_size=5
	_downsample_kernel_size=3
	_skip_connections=True
	layers=4

	if len(sys.argv)==2:
		inputdir = sys.argv[1]
	else:
		try:
			opts, args = getopt.getopt(argv,"hi:o:r:p:n:k:z:l:f:sd:",["indir=","outdir=","learning_rate=","input_features=","pretrained=","noise=","skip_connections","downsample_kernel_size"])
		except getopt.GetoptError:
			print('useage: autoencoder.py -i <inputdir> -o <outputdir> -k <kernelsize> -r <learningrate> -f <inputfeatures> -l <layers> -p <pretrained> -s -n <noise amount> -d <dropout> -z <latent depth>')
			print("\nexample invocation\n python3 autoencoder.py -i ../multi_input_test -k 5  -f 32 -z 256  -l 3")
			print("\treads images from ../multi_input_test, uses kernel size 5x5, 32 input features, 256 latent features, 3 layers")
			print("\tneeds images named foo_INPUT0.jpg,foo_INPUT1.jpg,foo_OUTPUT0.jpg , bar_INPUT0.jpg,bar_INPUT1.jpg etc")
			print("\twill train  (_INPUT0,_INPUT1) -> _OUTPUT0 ")
			print("")
			sys.exit(2)
		for opt, arg in opts:
			if opt == '-h':
				print('test.py -i <inputdir> -o <outputdir>')
				sys.exit()
			elif opt in ("-i", "--indir"):
				inputdir = makedir(arg)
			elif opt in ("-o", "--outdir"):
				outputdir = makedir(arg)
			elif opt in ("-r", "--learning_rate"):
				learningrate= float(arg)
			elif opt in ("-l", "--layers"):
				layers= int(arg)
			elif opt in ("-z", "--latent_depth"):
				_latent_depth= int(arg)
			elif opt in ("-s", "--skip_connections"):
				_skip_connections= True

			elif opt in ("-f", "--input_features"):
				_input_features= int(arg)

			elif opt in ("-d", "--downsample_kernel_size"):
				_downsample_kernel_size= int(arg)

			elif opt in ("-n", "--noise"):
				noise_amplitude= float(arg)
			elif opt in ("-n", "--dropout"):
				_dropout= float(arg)
			elif opt in ("-k", "--kernelsize"):
				_kernel_size=int(arg)

			elif opt in ("-p", "--pretrained"):
				print("setting pretrained model =",arg)
				pretrained = arg

	print("initializing device..")

	if not os.path.exists(outputdir):
		os.makedirs(outputdir)


	no_mps=True
	if torch.cuda.is_available(): device=torch.device("cuda")
	elif not no_mps and torch.backends.mps.is_available(): device = torch.device("mps")
	else: device = torch.device("cpu")
	 
	print("using device:",device.type)
	global g_device
	g_device=device

	print("grabbing dataset.."+inputdir)	
	dataloader=make_dataloader(inputdir,noise=noise_amplitude)
	io_channels=dataloader.dataset.io_channels()
	print("io channels=",io_channels)

	print("building model:")
#	ae = AutoEncoder(channels=[3,32,64,128,256],kernelSize= 7,skip_connections=True,skip_dropout=False)
#	ae = AutoEncoder(channels=[3,32,64,128,256],kernelSize= 5,skip_connections=False)
	encoder_ch,decoder_ch=make_layer_channel_list(io_channels,_input_features,_latent_depth,layers)

	net=EncoderDecoder(encoder_ch,decoder_ch,kernelSize= _kernel_size,skip_connections=_skip_connections,dropout=_dropout,downsample_kernel_size=_downsample_kernel_size)
	dump_state_dict(net)

	if pretrained !=None:
		print("loading pretrained model:",pretrained)
		load_model(net,pretrained)

	print("model config:",net.config_string())
	
	net.to(device)

#	for i,c in enumerate(ae.encoder_conv): print("layer[%d] is cuda?"%i,c.weight.is_cuda)
	#optimizer = torch.optim.SGD(ae.parameters(), lr=0.01)

	optimizer = [torch.optim.Adadelta(net.parameters(), lr=learning_rate,weight_decay=1e-8),
		torch.optim.Adadelta(net.parameters(), lr=learning_rate*0.25,weight_decay=1e-8)]

	print("Start training LR:", learning_rate,"\tsaving to:\t"+outputdir)

	save_freq=40 #todo save frequency based on time, once every 10 seconds raeehr than iterations..
	progress=Progress(outputdir)
	
	for i in range(0,50000):
		print("training epoch: ",i)
		train_epoch(device, net,optimizer[0],  dataloader,progress)
		
		if i%save_freq == 0:
			torch.save(net.state_dict(), outputdir+"my_trained_ae_"+str((i/save_freq)%8))

	print("training on some images.")
	torch.save(net.state_dict(), "my_trained_ae")

if __name__ == "__main__":
	main(sys.argv[1:])

		
