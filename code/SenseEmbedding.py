import numpy as np 
from scipy.stats import spearmanr
import pandas as pd 
import lxml.etree
from gensim.models import Word2Vec, FastText
import collections
import itertools
import matplotlib.pyplot as plt        
import re
import pickle
import matplotlib.cm as cm
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

############################################################################################################
#                                                                                                          #
#       Part 1: Extracting information from the xml file and saving it                                     #
############################################################################################################

lemmas_final = open('lemmas_fin.pickle', 'wb')
anchors_final = open('anchors_fin.pickle', 'wb')
synsets_final = open('synsets_fin.pickle', 'wb')
text_final = open('text_fin.pickle', 'wb')

#file path
path = '/home/diko/Desktop/eurosense.v1.0.high-coverage/EuroSense/eurosense.v1.0.high-coverage.xml'

anchors = []
lemmas = []
synsets = []
sentences = []

#Sysnset mappings in bablnet
mappings = []
mapp = open("resources/bn2wn_mapping.txt",'r')
for i in mapp:
    line = i.split()
    mappings.append(line[0])   


# extracting english data that we need
events = ('start','end')
#context = et.iterparse('./tester.xml',events=events)
anchor = []
lemma = []
synset = []
text = []

for index ,(event, element) in enumerate(tqdm(lxml.etree.iterparse(path,events=('start','end'), recover=True))):
    if event == 'start':
        
        if element.tag != 'sentence' and element.tag != 'annotations':
            if element.tag == 'text' and element.get('lang') == 'en':

                if element.text == None:
                    pass
                else:
                    text.append(element.text)
             #handling anchors, lemmas and synsets       
            elif element.tag == 'annotation' and element.get('lang') =='en':
                #we make a check if the synset is in bn 2 wn mapping file
                if element.text in mappings:
                    #if anchor list is empty(so this is the first elemenet on the actual iteration)
                    if  not anchor:
                        anchor.append(element.get('anchor'))
                        lemma.append(element.get('lemma'))
                        synset.append(element.text)
                    
                    else:
                        #If not the first element we check if it is part of a possible phrase on the previous element
                        #this way we reduce the number of synsets that we store
                        if element.get('anchor')  in anchor[len(anchor)-1]:
                            pass
                        else: 
                            anchor.append(element.get('anchor'))
                            lemma.append(element.get('lemma'))
                            synset.append(element.text)
                    
            
  #store the sentence, anchors for the sentence, lemmas and synsets and then making empty our variables
  # used to store the elements in our iteration
    if element.tag == 'sentence' and event == 'end':
        anchors.append(anchor)
        lemmas.append(lemma)
        synsets.append(synset)
        sentences.append(text)
        
        lemma = []
        synset = []
        anchor = []
        text = []
    element.clear()

#saving the extracted information into pickle structures
pickle.dump(synsets, synsets_final)
pickle.dump(lemmas, lemmas_final)
pickle.dump(anchors, anchors_final)
pickle.dump(sentences, text_final )




############################################################################################################
#                                                                                                          #
#       Part 2: Making dataset ready for the model and modeling                                            #
############################################################################################################


synsets = open('synsets_fin.pickle', 'rb')
synsets = pickle.load(synsets)

lemmas = open('lemmas_fin.pickle', 'rb')
lemmas = pickle.load(lemmas)

anchors = open('anchors_fin.pickle', 'rb')
anchors = pickle.load(anchors)

sentences = open('text_fin.pickle', 'rb')
sentences = pickle.load(sentences)


#Dropping the lemmas and anchors in each sentence if they are included in phrases, immideatly close to each other
# the goal is to mantain phrases instead of single unigram lemmas
for j in range(len(anchors)): 
    i = 0
    while i < len(anchors[j])-1:
        if i == 0:
            if anchors[j][i] in anchors[j][i+1]:
                anchors[j].remove(anchors[j][i])
                lemmas[j].remove(lemmas[j][i])
                synsets[j].remove(synsets[j][i])
            
                i = 0
            else:
                i = i+1
        elif anchors[j][i] in anchors[j][i+1] or anchors[j][i] in anchors[j][i-1]:
                anchors[j].remove(anchors[j][i])
                lemmas[j].remove(lemmas[j][i])
                synsets[j].remove(synsets[j][i])
            
                i = i-1
        else:
            i = i+1
#For phrase lemmas i join them in the format w1_w2_.._wn
    for i in lemmas[j]:
        nr_words = i.split()
        if len(nr_words) > 1:
            index = lemmas[j].index(i)
            lemmas[j][index]  = "_".join(nr_words)




#joining the lemmas in the formst lemma_synset (as required)
lemma_synset = []
for i in range(len(lemmas)):
    joined = []
    for j in range(len(lemmas[i])):
        joined.append(lemmas[i][j]+"_"+synsets[i][j])
    lemma_synset.append(joined)



#replacing anchors with lemma_synset
list_sentences = []
for idx, i in enumerate(tqdm(sentences)):
    if i: 
        new_sent = i[0]
        index  = sentences.index(i)
        for j in range(len(anchors[index])):
            if j == 0:
                a = anchors[index][j]
                rep = lemma_synset[index][j]
                new_sent = new_sent.replace(a+' ', " "+lemma_synset[index][j]+" ", 1)
            else:
                a = anchors[index][j]
                rep = lemma_synset[index][j]
                new_sent = new_sent.replace(' '+a+' ', " "+lemma_synset[index][j]+" ", 1)
        list_sentences.append(new_sent)


#saving the final corpus
with open('corpus_2.pickle', 'wb') as corp:
    pickle.dump(list_sentences, corp)


############################################################################################################
#                                                                                                          #
#       Part 2.2: Start modeling                                                                           #
############################################################################################################

file = open('corpus_2.pickle','rb')
list_sen1 = pickle.load(file)


#creating a corpus with tokens (words in our corpus) and transforming each token to lowercase
tokenizet_corpus = []

for idx, i in enumerate(tqdm(list_sen1)):
    sen = i.lower().split()
    tokenizet_corpus.append(sen)

# this is a piece of code which removes the unigrams (one lettter words) and punctuations

for i in tokenizet_corpus:
    for j in i:
        if len(j)<2:
            i.remove(j)
            
# this piece of code remove bugs from dataset of the following form @nbps_bn:1234567n
for i in tokenizet_corpus:
    for j in i:
        if not j[0].isalpha() and not j[0].isdigit():
            if '_bn:' in j:
                i.remove(j)

# initializing fasttext CBOW model
model = FastText(min_count=1,window=10, size=300 ,negative=20 , workers=3)

# initializing word2vec CBOW model
#model = Word2Vec(min_count=1,window=10, size=300 ,negative=20, workers=3)

# the following line loads the model which  i used for testin
#model = FastText.load('best_model.model')

#build vocabulary and start training
model.build_vocab(tokenizet_corpus)
model.train(tokenizet_corpus, total_examples=len(tokenizet_corpus), epochs=50, report_delay=1)

# save the embeddings vector
model.wv.save_word2vec_format('embeddings.vec', binary=False)

############################################################################################################
#                                                                                                          #
#       Part 3: evaluating the model                                                                       #
############################################################################################################

#first i take the data from the combined tab and create three separate arrays ot of it, word_1, word_2 and human for the human evaluation on word similarity

def testing_dataset()
    gold =[]
    myFile= open( "wordsim353/combined.tab", "r" )
    for aRow in myFile:
        gold.append(aRow.split('\t'))
    myFile.close()

    word_1 = []
    for i in gold:
        word_1.append(i[0])

    word_2 = []
    for i in gold:
        word_2.append(i[1])

    human = []
    for i in gold:
        human.append(i[2])
    return word_1, word_2, human


word_1, word_2, human = testing_dataset()

#taking model keys to find synsets for every word in the test file
keys= model.wv.vocab.keys()


# take all the senses for a specific word
def get_senses(word):   
    w1_senses = []
    word1 = word +'_bn:'
    for i in keys:
        if word1.lower() in i.lower() and len(i) == len(word)+13:
            w1_senses.append(i)
    return(w1_senses)

# create an array which contains the senses for every word by using the get_senses function
def create_sense_vector(word_1):
    w1_senses = []
    for i in word_1:
        senses = []
        senses = get_senses(i)
        w1_senses.append(senses)
    return w1_senses


w1_senses = create_sense_vector(word_1)

w2_senses = create_sense_vector(word_2)

# the first element of word senses are empty and the first element of human is a string so i delete them
del w1_senses[0]
del w2_senses[0]
del human[0]

#calculate the cosine similarity between every sense combination of two given words and return the highes score
def simmilarity(a,b):   
    score = 0
    for i in a:
        for j in b:
            sc = model.wv.similarity(w1=i, w2=j)
            if sc > score:
                score = sc
    print(score)
    return score


#calculate the similarities fo each word relying on the similarity function
def calculate_similarities(w1_senses, w2_senses):
    simm_vec=[]
    for i in range(len(w1_senses)):
        word1 = w1_senses[i]
        word2 = w2_senses[i]
        if word1 and word2:
            sim = simmilarity(word1, word2)
            simm_vec.append(sim)
        else:
            simm_vec.append(-1)
    print(simm_vec)
    return simm_vec


simmilarities = calculate_similarities(w1_senses, w2_senses)


for i in range(len(simmilarities)):
    simmilarities[i] = str(simmilarities[i])

    
for i in range(len(human)):
    human[i] = human[i].replace('\n',"")

#Calculate the spearman correlation
spearmanr(human, simmilarities)


############################################################################################################
#                                                                                                          #
#       Part 4:Visualizing plots with TSNE and pca                                                         #
############################################################################################################


# This piece of code was taken ready from the internet i just did some modification to turn it into functions

keys1 = ['love_bn:00087646v', 'love_bn:00090505v', 'love_bn:00052121n', 'love_bn:00090504v', 'love_bn:00083081v', 'love_bn:00000086n','sex_bn:00037634n', 'sex_bn:00070772n', 'sex_bn:00077747n', 'sex_bn:00082774v', 'sex_bn:00019285n']

def prepare_TSNE(keys):
    
    embedding_clusters = []
    word_clusters = []
    for word in keys1:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word, topn=10):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)



    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', n_iter=5000, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    return keys1, embeddings_en_2d, word_clusters

keys1, embeddings_en_2d, word_clusters = prepare_TSNE(keys1) 

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Similar words from Google News', keys1, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words_2.png')



# Testing if the embeddings are saved in the correct form

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/home/diko/Documents/Homework2/resources/embeddings.vec', binary=False)