from django.db import models


# Create your models here.
class Topic(models.Model):
    name = models.CharField(max_length=40)

    def __str__(self):
        return self.name


class Keywords(models.Model):
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE)
    name = models.CharField(max_length=40)

    def __str__(self):
        return self.name


class DeleteKeyWord(models.Model):
    name = models.CharField(max_length=124)
    topic = models.ForeignKey(Topic, on_delete=models.SET_NULL, blank=True, null=True)
    keywords = models.ForeignKey(Keywords, on_delete=models.SET_NULL, blank=True, null=True)

    def __str__(self):
        return self.name
