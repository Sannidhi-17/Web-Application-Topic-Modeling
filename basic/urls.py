from django.urls import path
from basic import views


urlpatterns = [
    path('', views.upload, name = 'basic'),

    # Urls for the interactive part on the webpage on the left-hand side
    path('changetopic/', views.change_topic, name = "change_topic"),  # Url for the change the number of topics
    path('addstopwords/', views.addstopwords, name = "add_stopwords"), # Url for the add stop words
    path('archievepage/', views.archieve, name = "archieve_page"), # Url for the archive page
    path('rename/',views.rename_topic, name = "rename_topic"), # Url for the rename the topic
    path('mergetopics/', views.merge_topics, name = "merge_topics"), # Url for the merge topics
    path('topresponses/', views.top_responses, name='top_Responses'), # Url for the top response
    path('removestopwords/', views.remove_stopwords, name='remove_stopwords'), # Url for the remove stop words
    path('delete/', views.delete_keywords, name='keyword_delete'), # Urls for the delete keywords
    path('ajax/load-keywords/', views.load_keywords, name='ajax_load_keywords'), # AJAX
    path('split/', views.split_topic, name = 'split_topic'),
     #Urls for downloading images and excel files
    path('down_file/', views.down_file),
    path('down_file_wordcloud/', views.down_file_wordcloud),
    path('down_file_Frequency_of_topics/', views.down_file_Frequency_of_topics),
    path('down_file_top_keywords/', views.down_file_top_keywords),
    path('down_file_Percentage_of_Topics/', views.down_file_Percentage_of_Topics),

]
