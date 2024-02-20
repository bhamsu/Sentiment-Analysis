""" Building Vectorizer/Embeddings models, and transforming text into vectors. """

# Importing all the required global as well local packages
from numpy import asarray, zeros
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class vectorizer:
    def __init__(self):
        pass

    @staticmethod
    def tfidf(train, test):
        # Implementing TfIdf Vectorizer using scikit-learn (sklearn)
        vect_tfidf = TfidfVectorizer()
        trainN = vect_tfidf.fit_transform(train)
        testN = vect_tfidf.transform(test)
        return trainN, testN

    @staticmethod
    def word_embed(train, test, maxLen = 100):
        word_tokenizer = Tokenizer()
        word_tokenizer.fit_on_texts(train)

        # Method texts_to_sequence converts the string data to its numeric form
        trainN = word_tokenizer.texts_to_sequences(train)
        testN = word_tokenizer.texts_to_sequences(test)

        # Vocab
        vocab_len = len(word_tokenizer.word_index) + 1
        print(f"Length of the vocabulary is {vocab_len}...")

        # Padding all the text into fixed length (maxLen) --> given
        trainN = pad_sequences(trainN, padding = 'post', maxlen = maxLen)
        testN = pad_sequences(testN, padding = 'post', maxlen = maxLen)

        # Loading GLoVe word embeddings and create an embeddings dictionary
        embeddings_dict = dict()
        glove = open('../models/a2_glove.6B.100d.txt', encoding = 'utf8')

        for line in glove:
            records = line.split()
            word = records[0]
            vector_dims = asarray(records[1:], dtype = 'float32')
            embeddings_dict[word] = vector_dims
        # closing the GLoVe file
        glove.close()

        # Creating Embedding Matrix having 100 columns
        # Containing 100-dimensional GloVe Word Embeddings for all words in our corpus

        embed_matrix = zeros((vocab_len, 100))
        for word, index in word_tokenizer.word_index.items():
            embed_vector = embeddings_dict.get(word)
            if embed_vector is not None:
                embed_matrix[index] = embed_vector
        print(f"Shape of the embeddings is {embed_matrix.shape}...")

        return trainN, testN, embed_matrix

    def __call__(self, *args, **kwargs):
        pass

    def __del__(self):
        pass