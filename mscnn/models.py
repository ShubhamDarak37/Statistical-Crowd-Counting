from django.db import models

# Create your models here.
class countPrediction(models.Model): 
    CrowdCount = models.IntegerField(default=0) 
    CDate = models.DateField(auto_now_add=False,default = '2001-01-01',null = True)
    CTime = models.TimeField(auto_now_add=False,default = '20:00',null =True)

class shopInfo(models.Model):
    areaId = models.IntegerField(primary_key=True)
    averageCount = models.IntegerField(default=0)
    monthAvgCount = models.IntegerField(default=0)
    dayAvgCount = models.IntegerField(default=0)
    CrowdLimit = models.IntegerField(default=0)
    

class areaOneMin(models.Model):
    areaId = models.ForeignKey(shopInfo,on_delete=models.CASCADE)
    areaCount = models.IntegerField()
    cDate = models.DateField(auto_now_add=False,default = '2001-01-01',null = True)
    cTime = models.TimeField(auto_now_add=False,default = '20:00',null =True)

class areaDay(models.Model):
    areaId = models.ForeignKey(shopInfo,on_delete=models.CASCADE)
    avg = models.IntegerField()
    min = models.IntegerField()
    max = models.IntegerField()
    cDate = models.DateField(auto_now_add=False,default = '2001-01-01',null = True)
    maxTime = models.TimeField(auto_now_add=False,default = '20:00',null =True)
    minTime = models.TimeField(auto_now_add=False,default = '20:00',null =True)

class crowdLimit(models.Model):
    areaId = models.ForeignKey(shopInfo,on_delete=models.CASCADE)
    cDate = models.DateField(auto_now_add=False,default = '2001-01-01',null = True)
    cTime = models.TimeField(auto_now_add=False,default = '20:00',null =True)
    areaCount = models.IntegerField()