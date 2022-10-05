import PIL
import os
from PIL import Image

#
#basewidth = 300


dirname="../training_images/any/"
for file in  os.listdir(dirname):
		img = Image.open(dirname+file)
		print("resizing:",file, "size=",img.size)
		img =img.resize((256,256), Image.ANTIALIAS)
		img.save(dirname+file)		

