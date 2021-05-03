from django import forms
from basic.models import DeleteKeyWord, Keywords;

# Form-1: uploading file, Analysis name and sheetname
class SheetName(forms.Form):
    sheetname = forms.CharField()
    output_file_name = forms.CharField()

# Form-2: Change the topic (collect the numbers of topic (integeres))
class ChangeTopic(forms.Form):
    change_topic = forms.CharField()

# Form-3: Collect the word which is not required
class Add_stopwords(forms.Form):
    add_stopwords = forms.CharField()

# Form-4: Collect the word which want to remove from the predefined list
class Remove_stopwords(forms.Form):
    remove_stopwords = forms.CharField()

# Form-5: Rename the topic name
class RenameTopic(forms.Form):
    topic_number = forms.CharField()
    name = forms.CharField()

# Form-6: Merge the two different topics
class MergeTopic(forms.Form):
    topic_number_1 = forms.CharField()
    topic_number_2 = forms.CharField()


class KeyWordDeletionForm(forms.Form):
    delete_keyword = forms.CharField()

class SplitTopic(forms.Form):
    topic_number = forms.CharField()
    topic_1_keywords = forms.CharField()
    topic_2_keywords = forms.CharField()

# Form-7: Change the number of top responses as per users requirement
class TopResponses(forms.Form):
    top_responses = forms.CharField()


# Form-8: Form for excluding the numbers from the analysis
class GeeksForm(forms.Form):
    geeks_field = forms.BooleanField()
