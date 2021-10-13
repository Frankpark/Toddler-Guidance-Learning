import cv2
import os, shutil
import json, uuid
from tqdm import tqdm

from pathlib import Path

rootdirname = "./image"
labeldirname = "bounding_data"
imagedirname = "raw_data"
rootoutdirname = "./output"
labelext = ".json"
dsize = (84,84)
W,H = 640,480


classnames = ["Ball","Boat","Cup","Fork","Glove","Hat","Shoe","Spoon","Tayo","Teddy"]
assert len(classnames) == 10

rootdir = Path(rootdirname)
labeldir = rootdir/labeldirname
imagedir = rootdir/imagedirname
rootoutdir = Path(rootoutdirname)


shutil.rmtree(rootoutdir, ignore_errors = True)  
os.mkdir(rootoutdir)

def xywh_to_ltrb(x,y,w,h):
    l = x 
    r = x + w
    t = y 
    b = y + h
    return (l,t,r,b)

for labelpath in tqdm(os.listdir(labeldir)) :
    with open(str(labeldir / labelpath)) as jsonfile :
        jsondata = json.load(jsonfile)
        
        imagepath = jsondata["data"]["image"]
        image = cv2.imread(imagepath)
        labels = jsondata["completions"][0]["result"]
        
        #print(labelpath)
        labelpathname, ext = os.path.splitext(labelpath)
        #outputdir = rootoutdir/imagepathname
        #os.makedirs(str(outputdir), exist_ok=True)
        #print(outputdir)

        for idx, label in enumerate(labels) :
            bbox =  label["value"]
            x, y, h, w = bbox["x"], bbox["y"], bbox["height"], bbox["width"]
            if w <= 0 or h <= 0 :
                continue # Ignore wrongly labeled width/heights
            name = bbox["rectanglelabels"][0]
            exists = False
            for candidate in classnames :
                if candidate in name :
                    name = candidate
                    exists = True
                    break
            if not exists :
                name = "None" 
            outputdir = rootoutdir / name
            os.makedirs(str(outputdir), exist_ok=True)

            l, t, r, b = xywh_to_ltrb(x,y,w,h)
            l,r = list(map((lambda x : x / 100.0 * W), (l,r)))
            t,b = list(map((lambda x : x / 100.0 * H), (t,b)))

            l,t = list(map(lambda x : max(0,int(x)), (l,t)))
            r = min(W, int(r))
            b = min(H, int(b))
            #l,t = list(map(lambda x : min(,int(x)), (l,t)))

            cropped_image = image[t:b,l:r]
            cropped_image = cv2.resize(cropped_image, dsize, interpolation=cv2.INTER_LINEAR)
            outputfilename = outputdir /(labelpathname + str(uuid.uuid4()) + "_" + str(idx) + ".png")
            cv2.imwrite( str(outputfilename), cropped_image)
            #cv2.rectangle(image,(l,t),(r,b),color=(255,0,0), thickness=2)
        
        #cv2.imwrite( str(outputdir /("target.png")), image)
        


        #print(image.shape)

