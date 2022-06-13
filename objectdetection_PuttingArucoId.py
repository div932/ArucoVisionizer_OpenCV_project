import cv2
import numpy as NumPy
import math

img = cv2.imread('CVtask.jpg')
#img= cv2.resize(img,(0,0),fx=0.5,fy=0.5)
cv2.imshow('vv',img)
cv2.waitKey(1000)
ORANGE_MIN = NumPy.array([0, 40, 90])
ORANGE_MAX = NumPy.array([27, 255, 255])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)
#edged = cv2.Canny(gray, 30, 150)
color={'green':[79,209,146],'orange':[9,127,240],'white':[210,222,228],'black':[0,0,0]}


contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
'''cv2.imshow('vv',thresh)
cv2.waitKey(3000)'''

i = 0
blank = NumPy.zeros(img.shape,NumPy.uint8)
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    if len(approx) == 4:
        
        x,y,w,h=cv2.boundingRect(approx)
        aspectratio = float(w)/h
        if aspectratio >=0.95 and aspectratio<=1.05:
            #print(approx)
            cord = [o[0].tolist() for o in approx]
            xmax=cord[0][0]
            xmin=cord[0][0]
            ymax=cord[0][1]
            ymin=cord[0][1]
            for i in cord:
                if i[0]>xmax:
                    xmax = i[0]
                if i[0]<xmin:
                    xmin = i[0]
                if i[1]>ymax:
                    ymax = i[1]
                if i[1]<ymin:
                    ymin = i[1]
            print(xmax,xmin,ymax,ymin)
            to1 = img[ymin:ymax,xmin:xmax]
            

                

            m1=(int((cord[0][0]+cord[1][0])/2),int((cord[0][1]+cord[1][1])/2))
            c=(int((cord[0][0]+cord[2][0])/2),int((cord[0][1]+cord[2][1])/2))
            if (m1[0]-c[0]) != 0:
                theta = math.atan((m1[1]-c[1])/(m1[0]-c[0]))
            else :
                theta = math.pi/(-2)
            print(theta*180/(math.pi))
            for i in color.keys():
                d = NumPy.array(color[i])
                d.reshape((3,))
                if (d==img[c[1],c[0],:]).any():
                    cv2.putText(img,i,c,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))

            print(img[c[1],c[0],:].shape)
            '''cv2.imshow('ff',t)
            cv2.waitKey(1000)'''
            cv2.circle(img,cord[0],5,(0,0,255),-1)
            cv2.drawContours(img,[approx], -1, (255, 0, 0), 3)
            print(cord)
       

    cv2.imwrite('cropped\\' + str(i) + '_img.jpg', img)


cv2.imwrite('new.jpg',img)
cv2.imshow('vv',img)
cv2.waitKey(0)

#putting aruco markers 
import cv2
from cv2 import BORDER_TRANSPARENT
from cv2 import BORDER_WRAP
from cv2 import BORDER_CONSTANT
from pyparsing import White
import cv2.aruco as aruco
import math
import numpy as NumPy
L=['Ha.jpg','HaHa.jpg','LMAO.jpg','XD.jpg']
iddict = {}

def arucoid(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f"DICT_5X5_250")
    arucodict = aruco.Dictionary_get(key)
    p = aruco.DetectorParameters_create()
    (c,i,r)= cv2.aruco.detectMarkers(img,arucodict,parameters=p)
    return (c,i,r)
def arucoprint(img):
    c,i,r=arucoid(img)
    topleft,topright,bottomright,bottomleft=arucocords(img)
    cx = int((topleft[0]+bottomright[0])/2)
    cy = int((topleft[1]+bottomright[1])/2)
    center = (cx,cy)
    cv2.putText(img,str(i),center,0,1,(0,0,0),1)

def arucocords(img):
    (c,i,r)=arucoid(img)
    if len(c)>0:
        i=i.flatten()
        for (markercorner,markerid) in zip(c,i):
            corner = markercorner.reshape((4,2))
            (topleft,topright,bottomright,bottomleft)=corner
            topleft = (int(topleft[0]),int(topleft[1]))
            topright = (int(topright[0]),int(topright[1]))
            bottomleft = (int(bottomleft[0]),int(bottomleft[1]))
            bottomright = (int(bottomright[0]),int(bottomright[1]))
        return topleft,topright,bottomright,bottomleft

def arucoangle(img):
    topleft,topright,bottomright,bottomleft=arucocords(img)
    cx = int((topleft[0]+bottomright[0])/2)
    cy = int((topleft[1]+bottomright[1])/2)
    px = int((topright[0]+bottomright[0])/2)
    py = int((topright[1]+bottomright[1])/2)
    m=(py-cy)/(px-cx)
    theta = math.atan(m)


    center = (cx,cy)
    cv2.circle(img,topright,5,(0,255,0),-1)
    cv2.circle(img,bottomright,5,(255,0,0),-1)
    cv2.circle(img,(0,0),5,(0,0,255),-1)
    cv2.imshow('ar',img)
    cv2.waitKey(3000)
    return center,(theta*180)/math.pi
def rotate_image(image, angle,center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=BORDER_CONSTANT,borderValue=(255,255,255))
    return result
def crp(img):
    #t = np.ones((600,600,3))
    '''cv2.imshow('gdff',t)
    cv2.waitKey(1000)'''
    topleft,topright,bottomright,bottomleft=arucocords(img)
    l=[topleft,topright,bottomright,bottomleft]
    xmax=l[0][0]
    xmin=l[0][0]
    ymax=l[0][1]
    ymin=l[0][1]
    for i in l:
        if i[0]>xmax:
            xmax = i[0]
        if i[0]<xmin:
            xmin = i[0]
        if i[1]>ymax:
            ymax = i[1]
        if i[1]<ymin:
            ymin = i[1]
    print(xmax,xmin,ymax,ymin)
    t = img[ymin:ymax,xmin:xmax]
    return t

y = cv2.imread('new.jpg')
c,i,r = arucoid(y)

print(i)
'''for i in L:
    x = cv2.imread(i)

    center,theta=arucoangle(x)
    f=rotate_image(x,(theta),center)
    df=crp(f)
    cv2.imshow('ar',f)
    cv2.waitKey(3000)
    print(df.shape)'''