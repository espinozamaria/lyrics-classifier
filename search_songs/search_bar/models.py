# search_bar/models.py
from django.db import models


class Song(models.Model):
    name = models.CharField(max_length=255)
    state = models.CharField(max_length=255)

    class Meta:
      verbose_name_plural = "song"

    def __str__(self):
        return self.name