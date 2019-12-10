# cities/urls.py
from django.urls import path
from django.conf.urls import url

from .views import HomePageView, SearchResultsView

urlpatterns = [
    path('get_song/<song_name>/<artist_name>/', HomePageView.as_view(), name='home'),
    path('get_lyrics/<track_id>/', SearchResultsView.as_view(), name='search_results')
]