# Generated by Django 3.1.7 on 2021-05-11 13:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mscnn', '0003_auto_20210511_1824'),
    ]

    operations = [
        migrations.AlterField(
            model_name='countprediction',
            name='CrowdCount',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='countprediction',
            name='StartTime',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
