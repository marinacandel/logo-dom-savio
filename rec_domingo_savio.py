import numpy as np
import cv2
def anonymize_face_pixelate(image, blocks=8):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)
    # return the pixelated blurred image
    return image
rostros =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ojos =cv2.CascadeClassifier('haarcascade_eye.xml')
upds = cv2.CascadeClassifier('cascade.xml')

cap=cv2.VideoCapture(0)
#cap=cv2.VideoCapture('nemvi.avi')


while True:

    ret, img=cap.read()
    #img variable de tipo mat -> array
    print('abriendo camara')
    

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('foto2',gray)

    blur=cv2.GaussianBlur(img,(5,5),0)
    #cv2.imshow('foto3',blur)

    borde = cv2.Canny(blur,50,150)
    #cv2.imshow('foto4',borde)

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst =cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    #cv2.imshow('foto5',dst)
    #se debe reducir el tamaño de la imagen 500x600
    caras = rostros.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=2,minSize=(20,20),maxSize=(400,400))
    for (x,y,w,h) in caras:
        cv2.rectangle(blur,(x,y),(x+w,y+h),(0,255,0),2)
        img_cortado=blur[y:y+h,x:x+w]
        #cv2.imshow('corte',img_cortado)
        anonymize_face_pixelate(img_cortado)
        ojitos = ojos.detectMultiScale(gray)
        for (ox,oy,ow,oh) in ojitos:
            cv2.rectangle(blur,(ox,oy),(ox+ow,oy+oh),(0,0,255),2)

    
    cv2.imshow('foto1',blur)
    print('mostrando')
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.realease()

cv2.destroyAllWindows()
    