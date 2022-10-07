import os
import sys,getopt
from typing import overload
from PIL import Image,ImageDraw,ImageFont,ImageColor
import colorsys
import math
import json
import random as rnd
import webbrowser

def main():
    print(sys.argv)
    if len(sys.argv)==1:
        print("specify directory of images to read")
        return
    autoopen=False
    for_training=False
    for a in sys.argv: 
        if a=="-a": autoopen=True
        if a=="-t": for_training=True
    label_set = []
    if len(sys.argv)>1:
        label_set=[x for x in sys.argv[2:] if x[0]!="-" ]
    if len(label_set)==0: 
        label_set=None
        print("no label set specified, using all labels [optionally supply label list in cmdline]")
    else: print("using label set:\n",label_set)


    annotated_images=render_labels_from_dir(sys.argv[1],2048,label_set,for_training)
    #["left/man","right/man","left/woman","right/woman"])
    print("compositing thumbnails: %d" % len(annotated_images))
    
    if len(annotated_images)>0:
        page=sys.argv[1]+"/html"
        result=make_thumbnail_grid(annotated_images,2048,pagename = page)
        print("at:",page+"/index.html")
    
        if autoopen:
            webbrowser.open("file:///"+os.getcwd()+"/"+page+"/index.html")
        else: print("created html view at "+page+"\nuse -a to automatically open when generating this")

    return

def render_labels_from_dir(dir,max=65536,label_set=None,for_training=False):
    if dir[-1]!="/":dir+="/"
    num_files=len(os.listdir(dir))
    imgls=[]
    num=len(os.listdir(dir))
    for_training=True
    training_image_size=(255,255)
    outindex=0
    training_dir=dir+"training/"
    if for_training:
        os.makedirs(training_dir,exist_ok=True)
    for filename in os.listdir(dir):
        ext=filename[filename.rfind("."):].upper()
        basename=filename[:filename.rfind(".")]
        if basename == "thumbnails": continue
        if not ext in [".JPG","JPEG","PNG"]: continue
        
        file =open(dir+basename+".json",)
        if not file: continue
        annotations=json.load(file) 
        file.close()

        if label_set:
            use_this=False
            for a in annotations:
                if a["validation"]["label"] in label_set:
                    use_this=True
            if not use_this:
                continue
        if len(imgls)>=max:
            print("limit of %d images reached, ignoring" % max)
            break
        image  = Image.open(dir+filename)
        if not (image.mode == 'RGB' or image.mode =='RGBA'):
            print("warning only RGB,RGBA supported mode=%s",image.mode)
            continue

        if len(imgls)%32 ==0:  print("render [%d/%d] %s" % (outindex,num,filename))

        labelled_polys=parse_annotations(annotations,label_set)
        if len(labelled_polys)==0: continue
        
        original_size=image.size
        if for_training:
            rendered=render_annotations_for_image((image,original_size),labelled_polys,255,0,unlabelled=(0,0,0,255),outline_thickness=0)
            image.resize(training_image_size)
            if image.mode=="RGB":
                image.save(training_dir+"image"+str(outindex)+"_INPUT.JPG")
                rendered.save(training_dir+"image"+str(outindex)+"_OUTPUT.JPG")
            else: print("warning todo rgba in training data")
            del image
            del rendered

        else:
            rendered=render_annotations_for_image((image,original_size),labelled_polys,96,224,outline_thickness=6)
            image.resize((128,128))
            imgls.append((basename,rendered))
        outindex+=1

    return imgls

def parse_annotations(annotations,label_set):
    labelled_polys={}

    num_broken=0
    for a in annotations:
        label=a["validation"]["label"]
        if label_set:
            if not label in label_set:
                continue
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
    
    if "left" in label: r,g,b=255,128,128
    elif "right" in label: r,g,b  = 128,255,128
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

def render_annotations_for_image(image_osize,annotations,fill_alpha=96,outline_alpha=224,unlabelled=(128,128,128,64),outline_thickness=1):
    image,original_size=image_osize

    overlay=Image.new(image.mode,image.size,unlabelled)

    dr=ImageDraw.Draw(overlay, 'RGBA')
    if unlabelled[3]>0:
        dr.rectangle( [0,0,image.width,image.height], unlabelled)

    sorted_polys=[[],[]]

    
    for label in annotations:
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

    if outline_alpha>0 and outline_thickness>0:
        foreach_poly(dr, poly_outline)

    del dr
    return composite 

def make_thumbnail_grid(images,width,min_image_width=128,pagename=None):
    imgsize=min_image_width
    if len(images)>0:
        while (len(images)*imgsize*imgsize*4 < width*width):
            imgsize*=2
    
    per_row=width//imgsize
    rows = (len(images)+per_row-1)//per_row
    
    grid=Image.new("RGBA",(width,rows*imgsize))

    thumbnails=[]
    print("putting %d images into grid" % len(images))
    
    for i,idimg in enumerate(images):
        ix=i% per_row
        iy=i//per_row
        id,img=idimg
        thumbnail = img.resize((imgsize,imgsize))
        thumbnails.append((id,thumbnail))

        grid.paste(thumbnail,(ix*imgsize, iy*imgsize))

    if pagename[-1]!="/": pagename+="/"


    if pagename:
        
        if not os.path.exists(pagename): os.mkdir(pagename)
        print("writing page at:",pagename)

        file  =open(pagename+"index.html","w")
        if file:
            print("file opened ok")
        file.write("<html><head></head><body>")
        file.write("<table>\n")
        for row in range(0,len(thumbnails)// per_row):
            print("row ",row)
            file.write("\t<tr>\n")
            for i in range(row*per_row,(row+1)*per_row):
                if i>len(thumbnails): break
                id,thumbnail=thumbnails[i]
                thumbname="thumbnail"+str(i)+".png";
                file.write("\t\t<td><a href='https://imagemonkey.io/annotate?mode=browse&view=unified&v=2&image_id=%s'><img src='%s'/></a><td>\n"
                % 
                    (
                    id,thumbname
                    )
                )
                thumbnail.save(pagename+thumbname)

            file.write("\t</tr>\n")

        file.write("</table>\n")
        file.write("</body></html>")
        file.close()

        

    return grid

main()