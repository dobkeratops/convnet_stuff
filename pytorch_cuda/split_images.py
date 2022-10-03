import PIL
import sys
import os
from PIL import Image

#
#basewidth = 300

# generate a test of recognisable input/output pairs
# INPUT0 = red,blue  INPUT1 = Green. 

dirname =sys.argv[1]
print("opening:",dirname)
if dirname[-1]!="/": dirname+="/"
for file in  os.listdir(dirname):
    if file[0] == '.': continue
    if not "_OUTPUT" in file: continue
    ext=file[file.rfind("."):]
    basename=file[0:file.rfind("_OUTPUT")]

    img = Image.open(dirname+basename+ext)
    
    
    print("resizing:",file, "size=",img.size)
    size=(256,256)
    img =img.resize(size, Image.ANTIALIAS)
    
    img1 = Image.new('RGB',size)
    img2 = Image.new('RGB',size)
    for x in range(0,size[0]):
        for y in range(0,size[1]):
            r,g,b=img.getpixel((x,y))
            img1.putpixel((x,y), (r,0,b))
            img2.putpixel((x,y), (0,g,0))
    print("output:",dirname+basename)
    img1.save(dirname+basename+"_INPUT0"+ext)
    img2.save(dirname+basename+"_INPUT1"+ext)