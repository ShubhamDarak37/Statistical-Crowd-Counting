from django.shortcuts import render,redirect
from .forms import *
from base64 import b64encode
# Create your views here.

# Create your views here.
from django.utils.timezone import datetime
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import MscnnConfig
from .models import *
from statistics import mean
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import Graph
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from datetime import datetime as dt,timedelta

def MSB(filters):
    """Multi-Scale Blob.
    Arguments:
        filters: int, filters num.
    Returns:
        f: function, layer func.
    """
    params = {'activation': 'relu', 'padding': 'same',
              'kernel_regularizer': l2(5e-4)}

    def f(x):
        #x1 = Conv2D(filters, 9, **params)(x)
        x2 = Conv2D(filters, 7, **params)(x)
        #x3 = Conv2D(filters, 5, **params)(x)
        x4 = Conv2D(filters, 3, **params)(x)
        x = concatenate([x2,x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
    return f

def MSCNN(input_shape):
    """Multi-scale convolutional neural network for crowd counting.
    Arguments:
        input_shape: tuple, image shape with (w, h, c).
    Returns:
        model: Model, keras model.
    """
    

    
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 9, activation='relu', padding='same')(inputs)
    x = MSB(4 * 16)(x)
    x = MaxPooling2D()(x)
    x = MSB(4 * 32)(x)
    #x = MSB(4 * 32)(x)
    #x = MaxPooling2D()(x)
    #x = MSB(3 * 64)(x)
    x = MSB(3 * 64)(x)
    x = Conv2D(1000, 1, activation='relu', kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(1, 1, activation='relu')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

size = 224
model = MSCNN((224, 224, 3))
model_graph = Graph()

model.load_weights('./model/final_weights.h5')

def home(request):
    areamin = areaOneMin.objects.all().filter(cDate=datetime.now())
    climit = crowdLimit.objects.all().filter(cDate = datetime.now())
    areaCount = shopInfo.objects.all().count()
    areacount = []
    for area in range(1,areaCount+1):
        areacount.append(area)
    return render(request, 'home.html',{'mincount':areamin,'areacount':areacount,'climit':climit})

def graphview(request):
    areamin = areaOneMin.objects.all().filter(cDate=datetime.now()).order_by('cTime')
    areaday = areaDay.objects.all().order_by('cDate')
    areaCount = shopInfo.objects.all().count()
    areacount = []
    for area in range(1,areaCount+1):
        areacount.append(area)
    return render(request,'graph.html',{'mincount':areamin,'daycount':areaday,'areacount':areacount})

def comparearea(request):
    areaCount = shopInfo.objects.all().count()
    Area = shopInfo.objects.all()
    area = []
    for areas in range(1,areaCount+1):
        area.append(areas)
    
    areacountday = []
    areaday = []
    areacountmonth = []
    areamonth = []
    for item in Area:
        areacountmonth.append(item.monthAvgCount)
        areaday.append(item.areaId)
        areacountday.append(item.dayAvgCount)
        areamonth.append(item.areaId)

    return render(request,'comparearea.html',{'areamonth':areamonth,'areaday':areaday,'areacountmonth':areacountmonth,'areacountday':areacountday})

def predict_count(request):
    if request.method == 'POST':
        print("hello")
        
        return render(request,'predicted_count.html')
        
    
    return render(request,'predict.html')

def calculate_avg(count_list,area,today):
    month = []
    day = []
    for item in count_list:
        day.append(item.areaCount)
        print(item.areaCount)
    dayavg = mean(day)
    areaDay.objects.filter(areaId_id=area,cDate__day=today.day,cDate__month=today.month,cDate__year=today.year).delete()
    areaDay.objects.create(areaId_id=area,avg=dayavg,min=0,max=0,cDate=today)
    areaday = areaDay.objects.all().filter(areaId_id=area,cDate__month=today.month)
    for item in areaday:
        month.append(item.avg)
        print(item.avg)
    monthavg = mean(month)
    shopInfo.objects.filter(areaId=area).update(monthAvgCount=monthavg)
    shopInfo.objects.filter(areaId=area).update(dayAvgCount=dayavg)

def predictImage(request):
    fileobj = request.FILES['filePath']
    fs = FileSystemStorage()
    filepathname = fs.save("video.mp4",fileobj)
    filepathname=fs.url(filepathname)
    testvideo = '.'+filepathname
    cap = cv2.VideoCapture(testvideo)
    btime = request.POST.get('ftime')
    bdate = request.POST.get('fdate')
    bdate = dt.strptime(bdate, '%Y-%m-%d')
    area = request.POST['farea']
    area = int(area)
    climit = shopInfo.objects.all().filter(areaId=area)
    print(climit)
    b = dt.strptime(btime, '%H:%M')
    second = 2
    images = []
    t = []
    seconds_added = timedelta(minutes=second)
    fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
    multiplier = fps * second
    success,image = cap.read()
    while success:
        frameId = int(round(cap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        success, image = cap.read()
        #print(frameId)
        if frameId % multiplier == 0:
            areaonemin = areaOneMin()
            images.append(image)
            areaonemin.cTime = b
            areaonemin.areaId_id = area
            b = b + seconds_added
            t.append(b)
            img = cv2.resize(image, (224, 224))
            img = img / 255.
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)[0][:, :, 0]
            dmap = cv2.resize(prediction,(224,224))
            dmap = cv2.GaussianBlur(prediction, (15, 15), 0)
            count = int(np.sum(dmap))

            if len(climit)>0:
                if count > 70:
                    crowdlimit = crowdLimit()
                    crowdlimit.areaId_id=area
                    crowdlimit.cDate=bdate
                    crowdlimit.cTime=b
                    crowdlimit.areaCount=count
                    crowdlimit.save()
                areaonemin.areaCount = count
                areaonemin.cDate = bdate
                areaonemin.save()
    today=bdate
    print(today.day)
    print(today.month)
    print(today.year)
    count_list = areaOneMin.objects.all().filter(areaId_id=area,cDate__day=today.day,cDate__month=today.month,cDate__year=today.year)
    if len(count_list)>0:
        calculate_avg(count_list,area,today)
    areaday = areaDay.objects.all().filter(areaId_id=area,cDate__month=today.month).order_by('cDate')
    return render(request,'predicted_count.html',{'mincount':count_list,'daycount':areaday})

#testimage = '.'+filepathname
#img = cv2.imread(testimage)
#countPred.InputImage=testimage
#img = cv2.resize(img, (224, 224))
#img = img / 255.
#img = np.expand_dims(img, axis=0)


#prediction = model.predict(img)[0][:, :, 0]
#dmap = cv2.resize(prediction,(224,224))
#dmap = cv2.GaussianBlur(prediction, (15, 15), 0)


#count = int(np.sum(dmap))
#countPred.CrowdCount=count
#context = {'filepathname':filepathname, 'count':count, 'countPred': countPred}

def arearegister(request):
    if request.method == 'POST':
        shopInfo.objects.all().delete()
        area_count = request.POST['fareas']
        climit = request.POST['climit']
        area_count = int(area_count)    
        for area in range(1,area_count+1):
            shopinfo = shopInfo()
            shopinfo.areaId = area
            shopinfo.crowdLimit = climit
            shopinfo.save()
        return redirect('/')

    else:
        return render(request, 'register.html')