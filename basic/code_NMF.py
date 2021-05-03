# Import all the libraries

import pandas as pd
import numpy as np
import os
import PyPDF2
import sys
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from nltk.tokenize import RegexpTokenizer
import libvoikko
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import gensim
from itertools import combinations
from nltk import flatten
from . import forms
import string

# Define the path for directory
os.chdir('../trailproject/media/')

# Pre-define stp_words for the Finnish language
stop_en = stopwords.words('finnish')


# Adding and removing the stop_words from the pre-define list of stop_words
class Stopwords:
    def __init__(self, add_stopwords='', remove_stopwords=''):
        self.add_stopwords = add_stopwords
        self.remove_stopwords = remove_stopwords
        self.stop_en = stop_en

    def adding_stopwords(self):
        print("The Size of original stopwords list:", len(self.stop_en))
        additional = self.stop_en
        print(additional)
        if self.add_stopwords != '':
            additional.append(self.add_stopwords)
        print("**--**---***")
        print(additional)

        print("The length of the additional stopwords list:", len(additional))
        b = int(len(additional))
        x = b - 235
        print("The number of extra words are ", x)

        print("the remove stopword is", self.remove_stopwords)
        if self.remove_stopwords != '':
            additional.remove(self.remove_stopwords)
        print("new stopword list is", additional)

        return additional


# Collect the Excel or PDF file file and convert into the .csv file
class generate_csvfile:
    def __init__(self, file_name, sheet_name=None):
        self.file_name = file_name
        self.sheet_name = sheet_name

    def convert_file(self):
        a = os.path.splitext(self.file_name)[0]
        f_name = a + '.csv'
        if self.file_name.endswith('.xlsx'):
            read_file = pd.read_excel(self.file_name, sheet_name=self.sheet_name)
            print(read_file)
            read_file.to_csv(f_name, index=None, header=True, encoding='utf-8-sig')
            print("This is an excel file")
            file = pd.read_csv(f_name, encoding='utf-8-sig', sep=',', error_bad_lines=False, names=['documents'])
            # print(file.head())

        elif self.file_name.endswith('.pdf'):
            pdf_file = self.file_name
            read_pdf = PyPDF2.PdfFileReader(pdf_file)
            number_of_pages = read_pdf.getNumPages()
            df1 = pd.DataFrame()
            # df1 = df1.append(pd.DataFrame(row, columns=['documents']),ignore_index=True)
            for page_number in range(number_of_pages):
                # use xrange in Py2
                page = read_pdf.getPage(page_number).extractText().split('.')
                # print(re.sub(r'([a-z])\n-([a-z])', r'\1', page[0]))
                # print(page)  # Extract page wise text then split based on spaces as required by you

                row = pd.Series(page)
                df1 = df1.append(pd.DataFrame(row, columns=['documents']), ignore_index=True)
                # df1 = df1.replace('\n',' ', regex=True)
                df1.documents = df1.documents.str.replace("\n-", "")
                df1.documents = df1.documents.str.replace("- \n", '')
                df1.documents = df1.documents.str.replace("-", '')
                df1.documents = df1.documents.str.replace("*", '')
                df1.documents = df1.documents.str.replace(".com", "/com", case=False)
                df1.to_excel('checking.xlsx')
                df1.documents = df1.documents.str.replace('[ˆ,†,˛,˝,˝,˜,š,˘,‹,ˇ,›,•,Ł,˙,",(,),<,>,{,},œ,”,”,•,*,*]', '')
                df1.documents = df1.documents.str.replace('[:,\n, @,˚,”]', ' ')
                # .documents = re.sub('([a-z])-(?=[a-z])', r'\1', df1.documents)
                # df1.documents = df1.documents.str.replace('[^\w-\w]', '\w\w')
                df1.documents = df1.documents.str.replace("ﬂ", "", case=False)
                df1['documents'].replace('', np.nan, inplace=True)
                df1.dropna(subset=['documents'], inplace=True)
                df1.to_csv(f_name, index=None, encoding='utf-8-sig')
                df1.to_excel('pdf to excel.xlsx')
                file = pd.read_csv(f_name, encoding='utf-8-sig', sep=',', error_bad_lines=False, names=['documents'])
                # print(file.head())
        else:
            print("Enter a valid file with extension")
            # return
            sys.exit()

        return file


# Apply pre-processing on the text, generate word cloud, count the coherence score and plot the coherence score
class Automate_topic_modeling:

    def __init__(self, dataframe, additional=stop_en):
        self.dataframe = dataframe
        self.stop_words = additional
        self.config = {
            "tfidf": {
                "sublinear_tf": True,
                "ngram_range": (1, 1),
                "max_features": 10000,
            },
            "nmf": {
                "init": "nndsvd",
                "alpha": 0,
                "random_state": 42,
            }
        }

        # Remove the numbers from the file

    def remove_numbers(self):
        self.dataframe['documents'] = self.dataframe['documents'].str.replace('\d+', '')

        return

        # Apply preprocessing

    def pre_processing(self):

        initial_df = self.dataframe
        # print(initial_df)
        # print("789456123")
        # initial_df = str(initial_df)
        initial_df['Index'] = np.arange(1, len(initial_df) + 1)
        initial_df = initial_df[['Index', 'documents']]
        initial_df['documents'] = initial_df['documents'].astype(str)
        new_df = pd.DataFrame(initial_df, index=initial_df.Index).stack()
        # new_df = pd.DataFrame(initial_df.documents.str.split('[.?!,]').tolist(), index=initial_df.Index).stack()
        new_df = new_df.reset_index([0, 'Index'])
        new_df.columns = ['Index', 'documents']
        new_df['documents'] = new_df['documents'].str.replace('[œ,Œ]', '-')
        new_df['documents'] = new_df['documents'].str.replace('ƒ⁄ﬁﬁ⁄', '')
        new_df['documents'] = new_df['documents'].str.replace('*', '')
        new_df['documents'] = new_df['documents'].str.lstrip()

        # # Remove empty row
        new_df['documents'].replace('', np.nan, inplace=True)
        new_df.dropna(subset=['documents'], inplace=True)
        # new_df.to_excel('checking.xlsx')
        # Capitalize the first letter
        new_df['documents'] = new_df['documents'].map(lambda x: x[0].upper() + x[1:])
        # new_df.to_excel('checking_upper.xlsx')
        # Converting into lower case
        # new_df['documents1'] = new_df.documents.map(lambda x: x.lower())
        new_df['documents1'] = new_df['documents'].str.replace(
            '[-,:,/,(,),",;,>,<,?,_,\n,❤,\t,??,ӻ,كw,큞,ԃ,ˮ,ĭ,ﬁﬁ,ﬂ,•,*,.,!]',
            '')
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\w]', '')
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\s]', ' ')
        new_df['documents1'] = new_df['documents1'].str.lstrip()
        # remove empty strings
        new_df['new_col'] = new_df['documents1'].astype(str).str[0]
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\w]', '')
        # new_df['documents1'] = new_df['documents1'].str.replace('[^\s]', ' ')
        nan_value = float("NaN")
        # Convert NaN values to empty string
        new_df.replace("", nan_value, inplace=True)
        new_df.dropna(subset=["new_col"], inplace=True)
        new_df.drop('new_col', inplace=True, axis=1)
        # Convert articles ino the tokens

        new_df['docuemnt_tokens'] = new_df.documents.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))

        # Apply Lemmatization (Voikko)
        # os.add_dll_directory(r'C:\Voikko')
        C = libvoikko.Voikko(u"fi")
        # C.setLibrarySearchPath("C:\Voikko")

        # Apply lemmatizations to the words
        def lemmatize_text(text):
            bf_list = []
            for w in text:
                voikko_dict = C.analyze(w)
                if voikko_dict:
                    bf_word = voikko_dict[0]['BASEFORM']
                else:
                    bf_word = w
                bf_list.append(bf_word)
            return bf_list

        new_df['lemmatized'] = new_df.docuemnt_tokens.apply(lemmatize_text)
        # new_df['documents'] = new_df['documents'].map(lambda x: [t for t in x if t not in self.stop_words])
        # stop_en = stopwords.words('finnish')
        new_df['article'] = new_df.lemmatized.map(lambda x: [t for t in x if t not in self.stop_words])
        # make sure the datatype of column 'article_removed_stop_words' is string
        new_df['article'] = new_df['article'].astype(str)
        new_df['article'] = new_df['article'].apply(eval).apply(' '.join)
        new_df['Index'] = np.arange(1, len(new_df['article']) + 1)
        new_df.to_excel('../static/assets/text_preprocessing.xlsx')
        return new_df

        # Generate the wordcloud

    def generate_wordcloud(self, new_df):
        text = new_df['article'].tolist()
        wordcloud = WordCloud(width=1600, height=800, background_color='black').generate(" ".join(text))
        # Open a plot of the generated image.
        plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        # plt.switch_backend('agg')
        plt.savefig('../static/images/wordCloud.png')
        return

        # Count the tf-idf matrix

    def tfidf_matrix(self, new_df):
        new_df['documents1'] = new_df['documents1'].str.replace('[^\w\s]', '')
        raw_documents = new_df['documents1'].tolist()
        self.count = new_df['documents1'].count()
        # print("the length of the dataframe is", self.count)
        # print(raw_documents)
        for i in range(len(raw_documents)):
            raw_documents[i] = raw_documents[i].lower()

        self.vectorizer = TfidfVectorizer(**self.config["tfidf"],
                                          stop_words=self.stop_words)
        A = self.vectorizer.fit_transform(raw_documents)
        terms = self.vectorizer.get_feature_names()

        return terms, A, raw_documents

        # Apply NMF-Model  model

    def nmf_model(self, A):
        kmin, kmax = 1, 1
        topic_models = []
        if self.count <= 100:
            kmin, kmax = 3, 10
        elif self.count <= 101 & self.count <= 500:
            kmin, kmax = 5, 15
        elif self.count > 501 & self.count <= 1000:
            kmin, kmax = 10, 30
        elif self.count > 1001 & self.count <= 3000:
            kmin, kmax = 10, 50
        else:
            kmin, kmax = 15, 70
        # try each value of k
        for k in range(kmin, kmax + 1):
            # print("Applying NMF for k=%d ..." % k)
            # run NMF
            model = decomposition.NMF(n_components=k, **self.config["nmf"])
            W = model.fit_transform(A)
            H = model.components_
            # store for later
            topic_models.append((k, W, H))
        return topic_models, H, W,

        # Building the word-to-vector dictionary

    def build_w2c(self, raw_documents):
        docgen = TokenGenerator(raw_documents, self.stop_words)
        new_list = []
        for each in docgen.documents:
            new_list.append(each.split(" "))
        new_list = [string for string in new_list if string != ""]
        # Build the word2vec model
        self.w2v_model = gensim.models.Word2Vec(vector_size=500, min_count=0.0005, sg=1)
        self.w2v_model.build_vocab(corpus_iterable=new_list)
        return self.w2v_model

        # Find the get descriptor

    def get_descriptor(self, all_terms, H, topic_index, top):
        # reverse sort the values to sort the indices
        top_indices = np.argsort(H[topic_index, :])[::-1]
        # now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:top]:
            top_terms.append(all_terms[term_index])
        return top_terms

        # Calculate the coherence score

    def get_coherence(self, terms, topic_models):
        k_values = []
        coherences = []
        dict = {}
        for (k, W, H) in topic_models:
            # Get all of the topic descriptors - the term_rankings, based on top 10 terms
            term_rankings = []
            for topic_index in range(k):
                term_rankings.append(self.get_descriptor(terms, H, topic_index, 10))
            # Now calculate the coherence based on our Word2vec model
            k_values.append(k)
            coherences.append(self.calculate_coherence(term_rankings))
            # print("K=%02d: Coherence=%.4f" % (k, coherences[-1]))
            # print("K=%02d: Coherence=%.4f" % (k, coherences[-1]))
            dict[k] = coherences[-1]
            # print(dict)
        newDict = {}
        # Iterate over all the items in dictionary and filter items which has even keys
        for (key, value) in dict.items():
            # Check if key is even then add pair to new dictionary
            if key % 2 == 0:
                newDict[key] = value

        max_key = max(newDict, key=newDict.get)
        # print(newDict)
        # print(max_key)
        return term_rankings, max_key, newDict

        # Calculate coherence score

    def calculate_coherence(self, term_rankings):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            # check each pair of terms
            pair_scores = []
            for pair in combinations(term_rankings[topic_index], 2):
                pair_scores.append(self.w2v_model.wv.similarity(pair[0], pair[1]))
            # get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            overall_coherence += topic_score
        # get the mean score across all topics
        return overall_coherence / len(term_rankings)

        # Plot the coherence score

    def plot_the_coherence_graph(pself, newDict):
        plt.figure(figsize=(12, 7))
        # create the line plot
        k_values = list(newDict.keys())
        coherences = list(newDict.values())
        # create the line plot
        plt.plot(k_values, coherences)
        plt.xticks(k_values, rotation=90)
        plt.xlabel("Number of Topics")
        plt.ylabel("Mean Coherence")
        # add the points
        plt.scatter(k_values, coherences)
        # find and annotate the maximum point on the plot
        ymax = max(coherences)
        xpos = coherences.index(ymax)
        best_k = k_values[xpos]
        plt.annotate("k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
        # show the plot
        # plt.show()
        plt.savefig('../static/images/coherenc_graph.png', transparent=True)
        return


# Using in the generating dictionary
class TokenGenerator:
    def __init__(self, documents, stopwords):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile(r"(?u)\b\w\w+\b")

    def __iter__(self):
        # print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall(doc):
                if tok in self.stopwords:
                    tokens.append("<stopword>")
                elif len(tok) >= 3:
                    tokens.append(tok)
            yield tokens


# Analyse the number topic numbers and visualize them
class topic_modeling:

    def __init__(self, tfidf, topic_numbers, feature_names, new_df):
        self.tfidf = tfidf
        self.topic_numbers = topic_numbers
        self.feature_names = feature_names
        self.new_df = new_df
        # print("----")
        # print(self.new_df)
        self.config = {
            "nmf": {
                "init": "nndsvd",
                "alpha": 0,
                "random_state": 42
            }
        }
        self.model = decomposition.NMF(n_components=self.topic_numbers, **self.config["nmf"])

        # Again count the NMF model on the specific number of topics

    def nmf_modeling(self):
        # apply the model and extract the two factor matrices
        W = self.model.fit_transform(self.tfidf)
        H = self.model.components_
        return H, W

        # Top 10 words dataframe topic_word

    def display_topics(self, no_top_words):
        col1 = 'topic'
        col2 = 'top_ten_words'
        dct = {col1: [], col2: []}
        for topic_idx, topic in enumerate(self.model.components_):
            dct[col1].append(int(topic_idx) + 1)
            dct[col2].append(", ".join([self.feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            x = pd.DataFrame.from_dict(dct)

        return x

        # Top 3 keywords themes per topic

    def top_three_keywords(self):
        # Themes top 3 keywords dataframe top_words1
        no_top_words = 3
        topic_word_3 = self.display_topics(no_top_words)
        topic_word_3['Themes'] = topic_word_3.top_ten_words.str.title()
        topic_word_1 = topic_word_3.loc[:, ['topic', 'Themes']]
        return topic_word_1

        # Find the responses which lies into which number of topics

    def documents_per_topic(self, W, topic_word_1):
        # print(W)
        df2 = pd.DataFrame({'topic': W.argmax(axis=1),
                            'documents': self.new_df['documents']},
                           columns=['topic', 'documents'])
        df2['documents'] = df2['documents'].apply(str).str.replace('\n', ' ')
        df2['documents'] = df2['documents'].apply(str).str.replace("/km", "€/km", case=False)
        df2['documents'] = df2['documents'].apply(str).str.replace(" gmail", "@gmail", case=False)
        # print(df2['documents'])
        # df2.to_excel('topic_number_with_responses.xlsx')
        # df2.to_pickle('topic.pkl')

        no_top_words = 10
        topic_word = self.display_topics(10)
        topic_word['documents'] = ''
        for i in range(self.topic_numbers):
            df3 = df2[df2['topic'] == i]
            x1 = df3['documents'].tolist()
            topic_word.iat[i, topic_word.columns.get_loc('documents')] = x1
            # i += 1
        topic_word_merge = pd.merge(topic_word_1, topic_word, on='topic')  # Merge two different dataframes and
        topic_word_merge['Length'] = topic_word_merge['documents'].str.len()
        return df2, topic_word_merge

        # Removve the topic after merging the topics

    def remove_row(self, topic_word_merge):
        topic_word_merge['Length'] = topic_word_merge['documents'].str.len()
        # print(topic_word_merge['Length'])
        topic_word_2 = topic_word_merge[topic_word_merge.Length != 0]
        topic_word_2['topic'] = np.arange(1, len(topic_word_2) + 1)
        topic_word_2 = topic_word_2.reset_index(drop=True)
        return topic_word_2

        # Frequency Plot

    def frequency_plot(self, topic_word_2):
        plt.figure(figsize=(30, 15))
        sns.set(font_scale=3)
        splot = sns.barplot(x="topic", y="Length", data=topic_word_2, color='blue')
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.0f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 10),
                           textcoords='offset points',
                           fontsize=25)
        plt.xlabel("Topics", size=30)
        plt.ylabel("Frequency", size=30)
        # path = self.create_directory
        plt.savefig('../static/images/Frequency_of_topic.png', transparent=True)
        return topic_word_2

        # Percentage Plot

    def percentage_plot(self, topic_word_2):
        topic_word_2['percent'] = (topic_word_2['Length'] / topic_word_2['Length'].sum()) * 100
        topic_word_2['percent'] = topic_word_2['percent'].astype(int)
        plt.figure(figsize=(30, 15))
        sns.set(font_scale=3)
        splot = sns.barplot(x="topic", y="percent", data=topic_word_2, color='blue')

        for p in splot.patches:
            splot.annotate('{:0.0f}%'.format(p.get_height()),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 10),
                           textcoords='offset points',
                           fontsize=25)

        # without_hue(splot, df.percentage_of_each_topic)
        plt.xlabel("Topics", size=30)
        plt.ylabel("percentage", size=30)
        # path = self.create_directory
        plt.savefig(
            '../static/images/percentage_of_topic.png',
            transparent=True)
        return

        # Top five responses per topic

    def get_top_snippets(self, all_snippets, W, topic_index, top):
        top = int(top)
        top_indices = np.argsort(W[:, topic_index])[::-1]
        # now get the snippets corresponding to the top-ranked indices
        top_snippets = []
        for doc_index in top_indices[0:top]:
            top_snippets.append(all_snippets[doc_index])
        return top_snippets

        # Create final csv file and save as excel file

    def final_output(self, topic_word_2, W, top=10):
        snippets = self.new_df['documents'].tolist()
        topic_word_2['responses'] = ''

        for i in range(0, self.topic_numbers):
            if topic_word_2['Length'][i] <= 10:
                y2 = topic_word_2['documents'][i]
                # print(y2)
                topic_word_2.at[i, 'responses'] = y2

            else:
                y1 = self.get_top_snippets(snippets, W, i, top)

                topic_word_2.at[i, 'responses'] = y1
        themes_keywords = topic_word_2
        # print(themes_keywords['top five documents'])
        themes_keywords = themes_keywords.rename(
            columns={'topic': 'Topic_id', 'Themes': 'Themes', 'top_ten_words': 'Top_ten_words',
                     'responses': 'Top_responses',
                     'documents': 'Documents',
                     'Length': 'Frequency_of_each_topic',
                     'percent': 'Percentage_of_each_topic'})

        themes_keywords.to_csv(
            '../static/assets/themes_keywords.csv',
            index=False)
        themes_keywords.to_excel(
            '../static/assets/themes_keywords.xlsx')
        return themes_keywords

        # For automatic subplot

    def printPFsInPairs(self):
        prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if self.topic_numbers in prime_numbers:
            self.topic_numbers = self.topic_numbers + 1
        for i in range(1, int(pow(self.topic_numbers, 1 / 2)) + 1):
            if self.topic_numbers % i == 0:
                self.n = i
                self.m = int(self.topic_numbers / i)

        # Keywords plot

    def plot_top_words(self, n_top_words, title):

        fig, axes = plt.subplots(self.m, self.n, figsize=(45, 45), tight_layout=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx + 1}',
                         fontdict={'fontsize': 50})
            ax.invert_yaxis()
            ax.tick_params(labelsize=50)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.savefig('../static/images/keywords.png',
                        transparent=True)

        return

        # Merge the different topics

    def fit_transform_merge(self, feature_1, feature_2, W_1, H_1):

        feature_1 = int(feature_1) - 1
        feature_2 = int(feature_2) - 1

        W = np.copy(W_1)
        H = np.copy(H_1)

        # merge (addition) column values of W
        W[:, feature_1] = W[:, feature_1] + W[:, feature_2]
        w = np.delete(W, feature_2, 1)
        # merge (addition) row values of H
        H[feature_1, :] = H[feature_1, :] + H[feature_2, :]
        self.model.components_ = np.delete(H, feature_2, 0)
        self.topic_numbers -= 1
        return w, H

        # Rename the topics

    def rename_topic(self, topic_number, new_name, themes_keywords):

        x = int(topic_number)
        y = themes_keywords.loc[themes_keywords['Topic_id'] == x]['Themes'].values[0]
        themes_keywords['Themes'] = themes_keywords['Themes'].replace(to_replace=y, value=new_name)
        themes_keywords.to_csv(
            '../static/assets/themes_keywords.csv',
            index=False)
        return themes_keywords

        # Remove the Keywords from the topic

    def split_topic(self, topic_number, topic_1_word, topic_2_word, themes_keywords):
        x = int(topic_number)
        same_topic_words = []
        same_topic_words.append(topic_1_word)
        same_topic_words = same_topic_words.replace(' ', ', ')
        other_topic_words = []
        other_topic_words.append(topic_2_word)
        other_topic_words = other_topic_words.replace(' ', ', ')
        S1 = set(same_topic_words)
        print(S1)
        S2 = set(other_topic_words)
        print(S2)
        y = [themes_keywords.loc[themes_keywords['Topic_id'] == x]['Top_ten_words'].values[0]]
        S3 = set(y)
        print(S1.issubset(S3))
        print(S2.issubset(S3))
        if S1.issubset(S3) & S2.issubset(S3) == True:
            print("Yes, Perfect")
        return themes_keywords
