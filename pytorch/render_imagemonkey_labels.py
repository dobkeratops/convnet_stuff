import os
import sys,getopt
from typing import overload
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
    
    annotated_images=render_labels_from_dir(sys.argv[1],4096,None,False)
    #["left/man","right/man","left/woman","right/woman"])
    print("compositing thumbnails:")
    result=make_thumbnail_grid(annotated_images,2048)
    result.show()

def render_labels_from_dir(dir,max=65536,filter_labels=None,show=False,):
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

        if not (image.mode == 'RGB' or image.mode =='RGBA'):
            print("warning only RGB,RGBA supported mode=%s",image.mode)
            continue
        labelled_polys=parse_annotations(annotations)
        if len(labelled_polys)==0: continue
        rendered=render_annotations_for_image(image,labelled_polys,96,224,outline_thickness=6,filter_labels=filter_labels)
        if show: rendered.show()
        
        imgls.append(rendered)
    return imgls

def parse_annotations(annotations):
    labelled_polys={}

    num_broken=0
    for a in annotations:
        label=a["validation"]["label"]
        for annotation in a["annotations"]:
            if annotation["type"]=='polygon':
                points=annotation["points"]

                vertices=[]
                broken=False
                for p in points:
                    x=p["x"]
                    y=p["y"]
                    if x is None or y is None:
                        broken=True
                    vertices.append((x,y))
                
                if not broken:
                    if not label in labelled_polys:
                        labelled_polys[label]=[]
                    labelled_polys[label].append(vertices)
                else: num_broken+=1
    if num_broken>0:
        print("warning found %d broken polys in %s" % (num_broken,annotations[0]["image"]["url"]))
    return labelled_polys

                
            #for k in val:
            #    print(k)

    
def color_of_label(label):
    r,g,b=colorsys.hsv_to_rgb(rnd.random()*6.248,1.0,1.0)
    r,g,b=int(r*255.0),int(g*255.0),int(b*255.0)

    # a few hardcoded , todo : specify coloration 
    if "left" in label: r,g,b=255,0,0
    elif "right" in label: r,g,b  = 0,255,0
    if "road" in label: r,g,b=255,0,255
    elif "pavement" in label: r,g,b  = 0,255,0
    return r,g,b

g_label_layer={}
g_background_labels=["road","pavement","foliage","bushes","tree","vegetation","sky","wall","kerbstone","wall","","mountains","cliffs","building","floor","ceiling","platform","path","paved","grass","soil","rock","sand","gravel","concrete"]

def get_label_layer(label):
    if label in g_label_layer:  return g_label_layer[label]

    layer=1
    for bgfrag in g_background_labels:
        if bgfrag in label:
            layer=0
            break
    
    g_label_layer[label]=layer
    return layer

def render_annotations_for_image(image,annotations,fill_alpha=96,outline_alpha=224,unlabelled=(128,128,128,64),outline_thickness=1,filter_labels=None):

    overlay=Image.new(image.mode,image.size,unlabelled)

    dr=ImageDraw.Draw(overlay, 'RGBA')
    if unlabelled[3]>0:
        dr.rectangle( [0,0,image.width,image.height], unlabelled)

    sorted_polys=[[],[]]

    
    for label in annotations:
        if filter_labels:
            if not label in filter_labels:
                continue
        sorted_polys[get_label_layer(label)].append( (label,annotations[label]) )

    def foreach_poly(dr,f):
        for poly_layer in sorted_polys:
            for label,polys in poly_layer:
                for poly in polys:
                    if len(poly)<3: continue
                    f(dr,label,poly)

    def poly_fill(dr,label,poly):
        r,g,b=color_of_label(label)
        dr.polygon(poly, (r,g,b,fill_alpha))

    def poly_outline(dr,label,poly):
        r,g,b=color_of_label(label)
        r=(r+255)//2; g=(g+255)//2; b=(b+255)//2
        dr.line(poly, (r,g,b,outline_alpha),width=outline_thickness ,joint=True)


    if fill_alpha>0:
        foreach_poly(dr,poly_fill)

    composite=Image.blend(image,overlay,unlabelled[3]*(1.0/255.0))

    del dr
    dr=ImageDraw.Draw(composite, 'RGBA')

    if outline_alpha>0:
        foreach_poly(dr, poly_outline)

    del dr
    return composite 

def make_thumbnail_grid(images,width,min_image_width=128):
    imgsize=min_image_width
    if len(images)>0:
        while (len(images)*imgsize*imgsize*4 < width*width):
            imgsize*=2
    
    per_row=width//imgsize
    rows = (len(images)+per_row-1)//per_row
    
    grid=Image.new("RGBA",(width,rows*imgsize))
    
    for i,img in enumerate(images):
        ix=i% per_row
        iy=i//per_row
        thumbnail = img.resize((imgsize,imgsize))
        grid.paste(thumbnail,(ix*imgsize, iy*imgsize))
    

    return grid

main()