# trabalho 

#Instale o gerenciador kaggle no ambiente do Colab e faça o upload do arquivo kaggle.json
!pip install -q kaggle
from google.colab import files
files.upload()  

 # Crie a pasta .kaggle
!rm -rf .kaggle
!mkdir .kaggle
!cp kaggle.json .kaggle/
!chmod 600 .kaggle/kaggle.json


!kaggle datasets download --force -d marlesson/news-of-the-site-folhauol

# Criar o DataFrame com os dados lidos diretamente da plataforma Kaggle
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

df = pd.read_csv("news-of-the-site-folhauol.zip")

# Atualizar o SPACY e instalar os modelos pt_core_news_lg
!pip install -U spacy
!python -m spacy download pt_core_news_lg
import spacy
nlp = spacy.load("news-of-the-site-folhauol.zip")

import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
	
	# Instalar os datasets stopwords, punkt e rslp do nltk
	
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')

# Carregar os módulos usados ao longo desse notebook

!pip install pyldavis &> /dev/null       // instalação do pyLDAvis

import warnings
warnings.filterwarnings('ignore')             # // ignora as advertencias 

# // modulos de machine learning e processamento 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np

# // nuvem de palavras

from wordcloud import WordCloud

# // visualização de dados

import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

from typing import List, Set, Any


SEED = 123


# Filtre os dados do DataFrame df e crie um DataFrame news_2016 que contenha apenas notícias de 2016 e da categoria mercado.

df['date'] = pd.to_datetime(df.date)
# Create a dataframe named news_2016


news_2016 = df[(df['data'].dt.year == 2016) & (df['categoria'] == 'mercado')]

print(news_2016.shape)
print(news_2016.head())

# NLTK Tokenizer and Stemmer

import nltk
nltk.download('rslp')
nltk.download('punkt')
news_2016.loc[:, 'nltk_tokens'] = news_2016.text.progress_map(tokenize)
from typing import List
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

def tokenize(text: str) -> List[str]:
    """
    Function for tokenizing using `nltk.tokenize.word_tokenize`
    Returns:
      - A list of stemmed tokens (`nltk.stem.RSLPStemmer`)
      IMPORTANT: Only tokens with alphabetic characters will be returned.
    """
    stemmer = RSLPStemmer()
    tokens = word_tokenize(text, language='portuguese')
    Alpha_tokens = [token for token in tokens if token.isAlpha()]
    stemmed_tokens = [stemmer.stem(token) for token in Alpha_tokens]
    return stemmed_tokens
	
	
	# Crie uma coluna spacy_doc que contenha os objetos spacy para cada texto do dataset de interesse. Para tal, carregue os modelos pt_core_news_lg e aplique em todos os textos 
	# // 9 casos
import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['text'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

import spacy
from tqdm import tqdm
tqdm.pandas()
news_2016['spacy_doc'] = news_2016['Mensagem'].progress_map(nlp)

# Carregue o modelo grande de português
nlp = spacy.load("pt_core_news_lg")

# Aplique o modelo em cada texto e crie a coluna 'spacy_doc'
news_2016['spacy_doc'] = news_2016['Mensagem'].map(nlp)

# Realize a Lematização usando SPACY

import spacy
nlp = spacy.load("pt_core_news_lg")
stopwords_list = stopwords()
lemmas_on_exclude = {"o", "em", "em o", "em a", "ano"}

def filter(token):
    return (
        token.is_Alpha and
        token.lemma_.lower() not in stopwords_list and
        token.lemma_.lower() not in lemmas_on_exclude
    )

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if filter(token)]

# Crie a coluna 'spacy_lemma' aplicando a função acima
news_2016['spacy_lemma'] = news_2016['Mensagem'].apply(lemmatize)

def stopwords() -> Set[str]:
    """
    Return complete list of stopwords
    """
    return set(list(nltk.corpus.stopwords.words("portuguese")) + list(STOP_WORDS))

complete_stopwords = stopwords()

def filter(w) -> bool:
    """
    Filter stopwords and undesired tokens
    """
    return (
        w.is_alpha and                
        not w.is_punct and            
        not w.is_space and            
        w.lemma_.lower() not in complete_stopwords 
    )

def lemma(doc) -> List[str]:
    """
    Apply spacy lemmatization on the tokens of a text

    Returns:
       - a list representing the standardized (with lemmatisation) vocabulary
    """ 
    return [w.lemma_.lower() for w in doc if filter(w)]
	
# Aplicando ao DataFrame
news_2016.loc[:, 'spacy_lemma'] = news_2016.spacy_doc.progress_map(lemma)


# Crie uma coluna tfidf no dataframe news_2016. Use a coluna spacy_lemma como base para cálculo do TFIDF. O número máximo de features que iremos considerar é 5000. E o token, tem que ter aparecido pelo menos 10 vezes (min_df) nos documentos.

from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self, doc_tokens: list):
        self.doc_tokens = doc_tokens
        self.tfidf = None

    def vectorizer(self):
        """
        Convert a list of tokens to tfidf vector
        Returns the tfidf vectorizer and attribute it to self.tfidf
        """
        # Junta os tokens de cada documento em uma string
        docs = [' '.join(tokens) for tokens in self.doc_tokens]
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            min_df=10,
            tokenizer=lambda x: x.split(), 
            preprocessor=None,
            token_pattern=None  
        )
        self.tfidf.fit(docs)
        return self.tfidf

    def __call__(self):
        if self.tfidf is None:
            self.vectorizer()
        return self.tfidf

# Gera a lista de tokens
doc_tokens = news_2016.spacy_lemma.values.tolist()
vectorizer = Vectorizer(doc_tokens)

def tokens2tfidf(tokens):
    tokens = ' '.join(tokens)
    array = vectorizer().transform([tokens]).toarray()[0]
    return array

news_2016.loc[:, 'tfidf'] = news_2016.spacy_lemma.progress_map(tokens2tfidf)


# Realize a extração de 9 tópicos usando a implementação do sklearn do algoritmo Latent Dirichlet Allocation. Como parâmetros, você irá usar o número máximo de iterações igual à 100 (pode demorar) e o random_seed igual a SEED que foi setado no início do notebook

from sklearn.decomposition import LatentDirichletAllocation

N_TOKENS = 9
SEED = 123

corpus = np.array(news_2016.tfidf.tolist())

lda = LatentDirichletAllocation(
    n_components=9,
    max_iter=100,
    random_state=123,
    learning_method='batch'
)
lda.fit(corpus)


# Crie uma coluna topic onde o valor é exatamente o tópico que melhor caracteriza o documento de acordo com o algoritmo de LDA.

def get_topic(tfidf: np.array):
    """
    Get topic for a lda trained model
    """
 
    topic_dist = lda.transform([tfidf])
    return int(np.argmax(topic_dist[0]))

news_2016['topic'] = news_2016.tfidf.progress_map(get_topic)

# Número de documentos vs tópicos

with sns.axes_style("ticks"):
    sns.set_context("talk")
    ax = news_2016['topic'].value_counts().sort_values().plot(kind = 'barh')
    ax.yaxis.grid(True)
    ax.set_ylabel("Tópico")
    ax.set_xlabel("Número de notícias (log)")
    sns.despine(offset = 10)
    ax.set_xscale("log")
	
# Crie uma nuvem de palavra para cada tópico.
# Use as colunas spacy_lemma e topic para essa tarefa.

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain

def plot_wordcloud(text: str, ax: plt.Axes) -> plt.Axes:
    """ Plota a wordcloud para o texto fornecido """
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return ax


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain

def plot_wordcloud(text: str, ax: plt.Axes) -> plt.Axes:
    """
    Plot the wordcloud for the text.
    Arguments:
        - text: string to be analised
        - ax: plt subaxis
    Returns:
        - ax
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return ax

fig, axis = plt.subplots(3, 3, figsize=(16, 12))
axis_  = axis.flatten()
for idx, ax in enumerate(axis_):
    ax_ = plot_wordcloud_for_a_topic(idx + 1, ax)
    if ax_ is None:
        plt.delaxes(ax)
        continue
    ax.set_title(f"Tópico {idx + 1}")
fig.tight_layout()
plt.show()



# Crie uma nuvem de entidades para cada tópico.
# Use as colunas spacy_lemma e topic para essa tarefa.

def plot_wordcloud_entities_for_a_topic(topic:int, ax:plt.Axes) -> plt.Axes:
    Topic_news = news_2016[news_2016['Topic'] == Topic]
    list_of_docs = Topic_news.spacy_ner.apply(lambda l : [w.replace(" ", "_") for w in l])
    list_of_words = chain(*list_of_docs)
    string_complete = ' '.join(list_of_words)
    if not len(string_complete):
        return None
    return plot_wordcloud(string_complete, ax)

fig, axis = plt.subplots(3, 3, figsize=(16, 12))

axis_  = axis.flatten()
for idx, ax in enumerate(axis_):
    ax_ = plot_wordcloud_entities_for_a_topic(idx + 1, ax)
    if ax_ is None:
        plt.delaxes(ax)
        continue
    ax.set_title(f"Tópico {idx + 1}")
fig.tight_layout()
