import os
import sys,getopt
from PIL import Image,ImageDraw,ImageFont,ImageColor
import colorsys
import math
import json
import random as rnd

def main():
    print(sys.argv)
    if len(sys.argv)==1:
        print("specify directory of images to read")
        return
    
    annotated_images=render_labels_from_dir(sys.argv[1],16384,False,None)
    #["left/man","right/man","left/woman","right/woman"])
    print("compositing thumbnails:")
    result=make_thumbnail_grid(annotated_images,8192)
    result.show()

def render_labels_from_dir(dir,max=16384,show=False,filter_labels=None):
    if dir[-1]!="/":dir+="/"
    num_files=len(os.listdir(dir))
    imgls=[]
    num=len(os.listdir(dir))
    for i,filename in enumerate(os.listdir(dir)):
        if i>max: break
        ext=filename[filename.rfind("."):].upper()
        basename=filename[:filename.rfind(".")]
        if not ext in [".JPG","JPEG","PNG"]: continue
        
        file =open(dir+basename+".json",)
        if not file: continue
        annotations=json.load(file) 
        file.close()
        image  = Image.open(dir+filename)
        if i%32 ==0:  print("render [%d/%d] %s" % (i,num,filename))

        labelled_polys=parse_annotations(annotations)
        if len(labelled_polys)==0: continue
        rendered=render_annotations_for_image(image,labelled_polys,64,(128,128,128,64),filter_labels)
        if show: rendered.show()
        
        imgls.append(rendered)
    return imgls

def parse_annotations(annotations):
    labelled_polys={}

    for x in annotations:
        label=x["validation"]["label"]
        for annotation in x["annotations"]:
            if annotation["type"]=='polygon':
                points=annotation["points"]

                vertices=[]
                for p in points:
                    vertices.append((p["x"],p["y"]))
                if not label in labelled_polys:
                    labelled_polys[label]=[]
                labelled_polys[label].append(vertices)

    return labelled_polys

                
            #for k in val:
            #    print(k)

    


def render_annotations_for_image(image,annotations,alpha=255,unlabelled=(128,128,128,0),filter_labels=None):
    
    overlay=Image.new('RGBA',image.size,(128,128,128,alpha))
    composite=Image.new('RGBA',image.size)

    dr=ImageDraw.Draw(overlay)
    
    for label in annotations:
        if filter_labels:
            if not label in filter_labels:
                continue
        polys=annotations[label]


        for poly in polys:
            
            if len(poly)<3: continue
            
            r,g,b=colorsys.hsv_to_rgb(rnd.random()*6.248,1.0,1.0)
            r,g,b=int(r*255.0),int(g*255.0),int(b*255.0)

            # a few hardcoded , todo : specify coloration 
            if "left" in label: r,g,b=255,0,0
            elif "right" in label: r,g,b  = 0,255,0
            if "road" in label: r,g,b=255,0,255
            elif "pavement" in label: r,g,b  = 0,255,0

            broken=False
            for x,y in poly:
                if x is None or y is None:
                    broken=True
            if broken: continue
            dr.polygon(poly, (r,g,b,alpha))
            dr.line(poly, (255,255,255,alpha), joint=True)

    composite.paste(image,(0,0))
    composite=Image.alpha_composite(composite,overlay)
    return composite

def make_thumbnail_grid(images,width,min_image_width=128):
    imgsize=min_image_width
    while (len(images)*imgsize*imgsize*4 < width*width):
        imgsize*=2
    
    per_row=width//imgsize
    rows = (len(images)+per_row-1)//per_row
    
    grid=Image.new("RGB",(width,rows*imgsize))

    
    for i,img in enumerate(images):
        ix=i% per_row
        iy=i//per_row
        thumbnail = img.resize((imgsize,imgsize))
        grid.paste(thumbnail,(ix*imgsize, iy*imgsize))
    

    return grid

main()