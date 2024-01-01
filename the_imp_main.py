import tensorflow as tf
import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import cv2
import os
from PIL import Image
from keras.models import Sequential
from keras_preprocessing.image import img_to_array
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import streamlit as st

def main():
    st.title('Devanagari text recognition :book:')
    st.markdown("<h2 style='text-align: center; color: grey;'>Please upload the image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["png","jpg","jpeg"])
    dig_btn = st.button('Digitize the text')
    if uploaded_file is not None:    
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    if dig_btn:
        if uploaded_file is None:    
            st.write("Invalid command, please upload an image")
        else:
            img = Image.open(uploaded_file)
            img = img.convert('L')
            img = np.array(img)
            #st.button('Digitize the text', Key = digitize)
            predict(img)

imp_characters = ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब',
                'भ','म','य','र','ल','व','श','ष','स','ह','क्ष','त्र','ज्ञ','अ','इ','उ','ऊ','ऋ','ए','क्र','ट्र','त्त','द्य',
                'प्र','श्र','रु','८']

d_characters = ['०','१','२','३','४','५','६','७','८','९']

u1_characters = ['े','ै','र्','ँ']

u2_characters = ['ो','ाै']

u_characters = ['ि','ी']

l_characters = ['ु','ू','्','ृ']

h_characters = ['क्','ख्','च्','ज्','ञ्','त्','थ्','ध्','न्','प्','फ्','ब्','भ्','म्','ल्','व्','श्','ष्','स्']

fc_characters = ['।','?']

string1=[]

mc_model = load_model(os.path.join('imp_Model', 'best_val_loss.hdf5'))

d_model = load_model(os.path.join('d_Model', 'best_val_loss.hdf5'))

u1_model = load_model(os.path.join('u1_Model', 'best_val_loss.hdf5'))
u2_model = load_model(os.path.join('u2_Model', 'best_val_loss.hdf5'))

lm_model = load_model(os.path.join('l_Model_2', 'best_val_loss.hdf5'))

hc_model = load_model(os.path.join('h_Model_2', 'best_val_loss.hdf5'))

fc_model = load_model(os.path.join('fi_Model_2', 'best_val_loss.hdf5'))

def prepare_img(window):
    shape1 = window.shape
    #print(shape1)
    if shape1[0]>shape1[1]:
        width = int((28 * shape1[1])/shape1[0])
        if width%2==0:
            width=width
        else:
            width=width+1
        height = 28
        a = int((32-width)/2)
        dim = (width, height)
        img = cv2.resize(window, dim)
        a = int((32-width)/2)
        outputImage = cv2.copyMakeBorder(img, 2, 2, a, a, cv2.BORDER_CONSTANT, value=0)
    else:
        height = int((28 * shape1[0])/shape1[1])
        if height%2==0:
            height=height
        else:
            height=height+1
        width = 28
        a = int((32-height)/2)
        dim = (width, height)
        img = cv2.resize(window, dim)
        a = int((32-height)/2)
        outputImage = cv2.copyMakeBorder(img, a, a, 2, 2, cv2.BORDER_CONSTANT, value=0)
    #img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #plt.figure(figsize=(2,2))
    #plt.imshow(outputImage)
    #plt.show()
    #print(outputImage.shape)
    return outputImage

def mc_recognition(window):
    img = prepare_img(window)
    X = img_to_array(img)
    X = np.expand_dims(X,axis =0)
    image1 = np.vstack([X])
    val=mc_model.predict(image1)
    position = np.argmax(val)
    string1.append(imp_characters[position])

def d_recognition(window):
    img = prepare_img(window)
    X = img_to_array(img)
    X = np.expand_dims(X,axis =0)
    image1 = np.vstack([X])
    val=d_model.predict(image1)
    position = np.argmax(val)
    string1.append(d_characters[position])

def u1_recognition(window):
    img = prepare_img(window)
    X = img_to_array(img)
    X = np.expand_dims(X,axis =0)
    image1 = np.vstack([X])
    val=u1_model.predict(image1)
    position = np.argmax(val)
    if position == 2:
        string1.insert(-1,u1_characters[position])
    else:
        string1.append(u1_characters[position])

def u2_recognition(window):
    img = prepare_img(window)
    X = img_to_array(img)
    X = np.expand_dims(X,axis =0)
    image1 = np.vstack([X])
    val=u2_model.predict(image1)
    position = np.argmax(val)
    string1.append(u2_characters[position])

def lm_recognition(window):
    img = prepare_img(window)
    X = img_to_array(img)
    X = np.expand_dims(X,axis =0)
    image1 = np.vstack([X])
    val=lm_model.predict(image1)
    position = np.argmax(val)
    string1.append(l_characters[position])

def hc_recognition(window):
    img = prepare_img(window)
    X = img_to_array(img)
    X = np.expand_dims(X,axis =0)
    image1 = np.vstack([X])
    val=hc_model.predict(image1)
    position = np.argmax(val)
    string1.append(h_characters[position])

def fc_recognition(window):
    img = prepare_img(window)
    X = img_to_array(img)
    X = np.expand_dims(X,axis =0)
    image1 = np.vstack([X])
    val=fc_model.predict(image1)
    position = np.argmax(val)
    string1.append(fc_characters[position])

def word_borders(here_img, thresh, bthresh=0.01):
    shape = here_img.shape
    check= int(bthresh*shape[0])
    image = here_img[:]
    top, bottom = 0, shape[0] - 1
    
    bg = np.repeat(thresh, shape[1])
    count = 0
    for row in range(1, shape[0]):
        if  (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            top = row - check
            break
    
    bg = np.repeat(thresh, shape[1])
    count = 0
    rows = np.arange(1, shape[0])
    for row in rows[::-1]:
        if  (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            bottom = row + count
            break

    d1 = (top - 2) >= 0 
    d2 = (bottom + 2) < shape[0]
    d = d1 and d2
    if(d):
        b = 2
    else:
        b = 0
    
    top = top
    return (top, bottom, b)

def word_preprocess(bgr_img):#gray image   
    blur = cv2.GaussianBlur(bgr_img,(5,5),0)
    th_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #converts black to white and inverse

    rows, cols = th_img.shape
    bg_test = np.array([th_img[i][i] for i in range(5)])
    if bg_test.all() == 0:
        text_color = 255
    else:
        text_color = 0
    
    tb = word_borders(th_img, text_color)
    lr = word_borders(th_img.T, text_color)
    dummy = int(np.average((tb[2], lr[2]))) + 2
    template = th_img[tb[0]:tb[1]+int(dummy/2), lr[0]:lr[1]]
    
    st.image(template, caption='Cropped and Preprocessed image')
    return (template, tb, lr)

def word_segmentation(prepimg):
    shape=prepimg.shape
    width = int((150 * shape[1])/shape[0])
    height = 150
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(prepimg, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    img1 = resized.T
    shape =  img1.shape
    bg = np.repeat(0, shape[1])
    array = []
    for row in range(1, shape[0]):
        if  (np.equal(bg, img1[row]).all()):
            array.append(row) 

    l = len(array)
    
    if l>4:
        array1=[0]
        array1.append(array[0])
        for i in range(0,l-2):
            if((array[i+1]-array[i])>10):
                array1.append(array[i])
                array1.append(array[i+1])
        array1.append(array[-1])

        array1.append(shape[0])
        segments = []

        leng=len(array1)
        shape = resized.shape
        x = 0
        while(x<leng-1):
            segment = resized[0:shape[0],array1[x]:array1[x+1]]
            segments.append(segment)
            x = x + 2
        return (segments, int(leng/2))
    else:
        return ([resized], 1)

def borders(here_img, thresh, bthresh=0.092):
    shape = here_img.shape
    check= int(bthresh*shape[0])
    image = here_img[:]
    top, bottom = 0, shape[0] - 1
    
    bg = np.repeat(thresh, shape[1])
    count = 0
    for row in range(1, shape[0]):
        if  (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            top = row - check
            break
    
    bg = np.repeat(thresh, shape[1])
    count = 0
    rows = np.arange(1, shape[0])
    
    for row in rows[::-1]:
        if  (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            bottom = row + count
            break

    d1 = (top - 2) >= 0 
    d2 = (bottom + 2) < shape[0]
    d = d1 and d2
    if(d):
        b = 2
    else:
        b = 0
    
    top = top
    
    return (top, bottom, b)

def preprocess(bgr_img):#gray image   
    blur = cv2.GaussianBlur(bgr_img,(5,5),0)
    th_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] #converts black to white and inverse

    rows, cols = th_img.shape
    bg_test = np.array([th_img[i][i] for i in range(5)])
    if bg_test.all() == 0:
        text_color = 255
    else:
        text_color = 0
    
    tb = borders(th_img, text_color)
    lr = borders(th_img.T, text_color)
    dummy = int(np.average((tb[2], lr[2]))) + 2
    template = th_img[tb[0]:tb[1]+int(dummy/2), lr[0]:lr[1]]
    
    return (template, tb, lr)

def segmentation(bordered, siro_rekha, width):
    shape = bordered.shape
    img = bordered
    if siro_rekha>15:
        img[siro_rekha-8:siro_rekha+10,0:width] = 0
    else:
        img[0:siro_rekha+13,0:width] = 0
    image = img.T
    shape = image.shape

    bg = np.repeat(0, shape[1])
    array = [0]
    for row in range(1, shape[0]):
        if  (np.equal(bg, image[row]).all()):
            array.append(row)

    l1 = len(array)
    
    if l1>0:
        array1=[]
        
        for i in range(0,l1-2):
            if((array[i+1]-array[i])>=6):
                array1.append(array[i])
                array1.append(array[i+1])
        
        l2 = len(array1)
        for i in range(0,l2-2):
            if((array1[i+1]-array1[i])<=2):
                array1[i]=-1
                array1[i+1]=-1
        
        array2 = []
        for i in range(0,l2):
            if(array1[i]!=-1):
                array2.append(array1[i])
        array2[-1]=array2[-1]+5
        if((l2%2!=0)):
            array2.pop(1)
        #print(array2)
        segments = []
        leng=len(array2)
        #print(leng)
        main_img = cv2.imread("resized.jpg",0)
        shape = main_img.shape
        x = 0
        while(x<leng-1):
            if((array2[x+1]-array2[x])<20):
                segment = main_img[0:shape[0],array2[x]-5:array2[x+1]+5]
                segments.append(segment)
                x = x + 2
            else:
                segment = main_img[0:shape[0],array2[x]:array2[x+1]+3]
                segments.append(segment)
                x = x + 2
        return segments
    else:
        return [main_img]
    
def half_letter_segmentation(window):
    shape1 = window.shape # for checking whether the middle letter have joint half letter
    if (shape1[1]>=int(1.6*shape1[0])): # the middle letter have joint half letter
        with_character = []
        for column in range(int(0.3*shape1[1]),int(shape1[1]-0.3*shape1[1])):
            k = 0
            for row in range(int(0.2*shape1[0]),shape1[0]):
                if window[row][column]==255:
                    k+=1
                else:
                    k=k
            with_character.append(k)
        l = len(with_character)
        test = []
        for g in range(l-10):
            f = with_character[g+9]+with_character[g+8]+with_character[g+7]+with_character[g+6]+with_character[g+5]
            -with_character[g+4]-with_character[g+3]-with_character[g+2]-with_character[g+1]-with_character[g]
            test.append(f)
        p=np.argmax(test)

        window_x = window[0:shape1[0],0:int(p+0.3*shape1[1]+3)]
        window_y = window[0:shape1[0],int(p+0.3*shape1[1]+3):shape1[1]]

        hc_recognition(window_x)

        mc_recognition(window_y)
        return(window_x,window_y)
    elif(shape1[1]<int(0.45*shape1[0])):
        window=window
        string1.append('ा')
        return(window)
    else: # the middle letter doesnot have any half letter joined
        window=window
        mc_recognition(window)
        return(window)

def character_segmentation(segments, siro_rekha, low_level, lowest_level, average_low):         
    i = 0
    if siro_rekha<=15: #there is no upper modifier in the word
        for simg in segments: #taking in segmented joint letter
            shape = simg.shape
            if low_level[i] > average_low: #there is lower modifier in the letter
                window0 = simg[0:lowest_level,0:shape[1]]
                half_letter_segmentation(window0)
                window1 = simg[lowest_level:shape[0],0:shape[1]]
                lm_recognition(window1)
            else: #there is no lower modifier in the letter
                window0 = simg[0:low_level[i],0:shape[1]]
                half_letter_segmentation(window0)
            i+=1
    else: #there is upper modifier in the word 
        for simg in segments: #taking in segmented joint letter
            shape = simg.shape
            a = 0
            for column in range(1, shape[1]): #for checking if there is upper modifier in the letter
                if (simg[int(siro_rekha/2)][column]==255):
                    a +=1
                else:
                    a = a
            #print("a =",a)
            touching_points=[]
            if low_level[i] > average_low: #there is lower modifier in the letter
                if a>0: #there is upper modifier in the letter 
                    window0 = simg[siro_rekha-5:lowest_level,0:shape[1]]
                    half_letter_segmentation(window0)
                    window1 = simg[lowest_level:shape[0],0:shape[1]]
                    window2 = simg[0:siro_rekha-3,0:shape[1]]
                    lm_recognition(window1)
                    u1_recognition(window2)
                else: #there is no upper modifier in the letter
                    window0 = simg[siro_rekha-5:lowest_level+3,0:shape[1]]
                    half_letter_segmentation(window0)
                    window1 = simg[lowest_level:shape[0],0:shape[1]]
                    lm_recognition(window1)
            else: #there is no lower modifier in the letter
                if a>0: #there is upper modifier in the letter
                    for column in range(1,shape[1]): #for knowing whether the upper modifier is single touching or double touching
                        if (simg[siro_rekha-8][column]==255):
                            touching_points.append(column)
                    if(len(touching_points)==1):
                        temp = touching_points[0]+1
                        touching_points.append(temp)
                    
                    front_tp = touching_points[0]
                    back_tp = touching_points[-1]
                    max_diff = np.diff(touching_points).max()
                    for num in range(0,len(touching_points)-1):
                        if (touching_points[num+1]-touching_points[num])== max_diff:
                            x = touching_points[num]
                            y = touching_points[num+1]
                    
                    tcs=[]
                    if max_diff>=15: # the upper modifier is two touching upper modifier
                        for sub in range(2,12):
                            t=0
                            for row in range(siro_rekha+13,lowest_level): # for checking whether the middle level portion of the upper modifier is in front or back
                                if (simg[row][y-sub]==255):
                                    t+=1
                                else:
                                    t=t
                            tcs.append(t)
                        
                        test=1
                        for num in range(10):
                            if tcs[num]==0:
                                test=0
                                rem=num
                            else:
                                test=test
                        if test==1: # the middle level portion of the upper modifier is at front
                            window0 = cv2.imread("temp_"+str(i)+".jpg",0)
                            window0 = window0[siro_rekha:low_level[i],x+5:shape[1]]
                            half_letter_segmentation(window0)
                            window1 = simg[0:lowest_level,0:back_tp+10]
                            window1[siro_rekha+3:lowest_level,x+3:back_tp+10]=0
                            
                            string1.append('ि')
                        else: # the middle level portion of the upper modifier is at the back
                            window0 = cv2.imread("temp_"+str(i)+".jpg",0)
                            window0 = window0[siro_rekha-5:low_level[i],0:y-rem-2]
                            half_letter_segmentation(window0)
                            window1 = simg[0:lowest_level,0:back_tp+10]
                            window1[siro_rekha+3:lowest_level,0:y-rem-2]=0
                            window1 = window1[0:lowest_level,int(shape[1]/3):back_tp+10]
                            
                            string1.append('ी')
                    else: # the upper modifier is a single touching modifier
                        for sub in range(2,12):
                            t=0
                            for row in range(siro_rekha+13,lowest_level): # for checking whether the upper modifier has connected middle level or not
                                if (simg[row][front_tp-sub]==255):
                                    t+=1
                                else:
                                    t=t
                            tcs.append(t)    
                        
                        test=1
                        for num in range(10):
                            if tcs[num]==0:
                                test=0
                                rem=num
                            else:
                                test=test
                        if test==1: # the upper modifier has no connected middle level
                            window0 = simg[siro_rekha-5:low_level[i],0:shape[1]]
                            half_letter_segmentation(window0)
                            window1 = simg[0:siro_rekha-3,0:shape[1]]
                            
                            u1_recognition(window1)
                        else: # the upper modifier has connected middle level
                            window0 = cv2.imread("temp_"+str(i)+".jpg",0)
                            window0 = window0[siro_rekha-5:low_level[i],0:front_tp-rem-2]
                            half_letter_segmentation(window0)
                            window1 = simg[0:lowest_level,0:back_tp+10]
                            window1[siro_rekha+8:lowest_level,0:front_tp-rem-2]=0
                            window1 = window1[0:lowest_level,int(shape[1]/3):back_tp+10]
                            
                            u2_recognition(window1)
                else: #there is no upper modifier in the letter
                    window0 = simg[siro_rekha-10:low_level[i],0:shape[1]]
                    half_letter_segmentation(window0)
            i+=1

def siro_rekha_finder(resized):
    photo = resized
    shape=photo.shape
    number = []

    for row in range(1, int(0.6*shape[0])):
        count = 0
        for column in range(1, shape[1]):
            if (photo[row][column]==255):
                count +=1
            else:
                count = count
        number.append(count)
  
    siro_rekha=np.argmax(number)+2
    if (number[siro_rekha-2]<int(0.4*shape[1])):
        return 0
    else:
        return siro_rekha

def find_low_level(segments):
    thresh = 255
    low_level = []
    a = 0
    for simg in segments:
        shape = simg.shape
        check= int(0.1*shape[0])
        image = simg[:]
        bottom = shape[0] - 1

        bg = np.repeat(thresh, shape[1])
        count = 0
        rows = np.arange(1, shape[0])
        #print(rows)
        for row in rows[::-1]:
            if  (np.equal(bg, image[row]).any()) == True:
                count += 1
            else:
                count = 0
            if count >= check:
                bottom = row + count
                break
   
        low_level.append(bottom)
        a+=1

    lowest_level = np.min(low_level)
    range_lp = np.max(low_level)-lowest_level
    if range_lp<12:
        average_low=np.max(low_level)
    else:
        average_low = int(np.average(low_level))
    return (low_level, lowest_level, average_low)

def predict(image): 
    string1.clear()
    img = image
    prepimg, tb, lr = word_preprocess(img)
    words, num= word_segmentation(prepimg)
    count = 0
    for simg in words:
        if(count==num-1):
            prepimg, tb, lr = preprocess(simg)
            shape = prepimg.shape
            if(shape[1]>int(1.2*shape[0])):
                width = int((100 * shape[1])/shape[0])
                height = 100
                dim = (width, height)
                resized = cv2.resize(prepimg, dim, interpolation = cv2.INTER_AREA)
                resized = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                cv2.imwrite("resized.jpg", resized)
                siro_rekha = siro_rekha_finder(resized)
                if (siro_rekha==0):
                    d_recognition(prepimg)
                    count+=1
                else:
                    segments=segmentation(resized, siro_rekha, width)
                    low_level, lowest_level, average_low = find_low_level(segments)
                    i=0
                    for simg in segments:
                        shape = simg.shape
                        cv2.imwrite("temp_"+str(i)+".jpg",simg)
                        i+=1
                    character_segmentation(segments, siro_rekha, low_level, lowest_level, average_low) 
                    os.remove("resized.jpg")
                    i = 0
                    for simg in segments:
                        os.remove("temp_"+str(i)+".jpg")
                        i+=1
                    count+=1
            else:
                fc_recognition(prepimg)
                count+=1
        else:
            prepimg, tb, lr = preprocess(simg)
            shape = prepimg.shape
            width = int((100 * shape[1])/shape[0])
            height = 100
            dim = (width, height)
            resized = cv2.resize(prepimg, dim, interpolation = cv2.INTER_AREA)
            resized = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            cv2.imwrite("resized.jpg", resized)
            siro_rekha = siro_rekha_finder(resized)
            if (siro_rekha==0):
                d_recognition(prepimg)
                count+=1
            else:
                segments=segmentation(resized, siro_rekha, width)
                low_level, lowest_level, average_low = find_low_level(segments)
                i=0
                for simg in segments:
                    shape = simg.shape
                    cv2.imwrite("temp_"+str(i)+".jpg",simg)
                    i+=1
                character_segmentation(segments, siro_rekha, low_level, lowest_level, average_low)
                os.remove("resized.jpg")
                i = 0
                for simg in segments:
                    os.remove("temp_"+str(i)+".jpg")
                    i+=1
                string1.append(' ')
                count+=1
    
    string= ''.join(string1)
    st.write(string)

if __name__ == "__main__":
    main()