# Generated by Django 3.1.7 on 2021-05-22 18:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mscnn', '0006_areaday_areaonemin_shopinfo'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='areaday',
            name='cTime',
        ),
    ]
