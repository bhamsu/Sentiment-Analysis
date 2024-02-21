""" Pre-processing of the text data is performed here. """

# Importing all the required global as well local packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

class process_text_data:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        # print(self.raw_data)
        nltk.download('stopwords')
        self.port_stem = PorterStemmer()
        self.snow_stemmer = SnowballStemmer(language = 'english')   # this stemmer requires a language parameter

    def drop_nan(self):
        # Checking and removing all the null values from dataset
        if self.raw_data.isnull().sum()['text'] > 0 or self.raw_data.isnull().sum()['target'] > 0:
            print("Null values are found...")
            print(self.raw_data.isnull().sum())
            self.raw_data.dropna()
            print("All Null values are removed...")
        else:
            print("Zero Null values are found...")
        self.raw_data.replace({'target':{4:1}}, inplace = True)
        return 0

    @staticmethod
    def lower(data):
        # Lower casing all the characters in the text given
        if not data.islower():
            return data.lower()
        return data

    @staticmethod
    def rm_htmlTags(text):
        # Removing HTML tags: replaces anything between opening and closing <> with empty space
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    def porter_stemming(self, content):
        # Stemming: It is a process of transforming each word into its Root word
        # example: actor, actress, acting -> act

        stemmed_content = re.sub('[^a-zA-Z]', ' ', content) # Removing Punctuations and numbers
        stemmed_content = self.rm_htmlTags(stemmed_content) # Removing HTML Tags
        stemmed_content = self.lower(stemmed_content)       # Lower Casing all the words
        stemmed_content = re.sub(r"\s+[a-zA-Z]\s+", ' ', stemmed_content)  # Removing Single characters
        stemmed_content = re.sub(r'\s+', ' ', stemmed_content)  # Removing multiple spaces

        # Removing stopwords and performing stemming
        stemmed_content = stemmed_content.split()
        stemmed_content = [self.port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

    def snowball_stemming(self, content):
        # Stemming: It is a process of transforming each word into its Root word
        # example: actor, actress, acting -> act

        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # Removing Punctuations and numbers
        stemmed_content = self.rm_htmlTags(stemmed_content)  # Removing HTML Tags
        stemmed_content = self.lower(stemmed_content)  # Lower Casing all the words
        stemmed_content = re.sub(r"\s+[a-zA-Z]\s+", ' ', stemmed_content)  # Removing Single characters
        stemmed_content = re.sub(r'\s+', ' ', stemmed_content)  # Removing multiple spaces

        # Removing stopwords and performing stemming
        stemmed_content = stemmed_content.split()
        stemmed_content = [self.snow_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

    def __call__(self, *args, **kwargs):
        self.drop_nan()
        print("Starting stemming operations...")
        self.raw_data['stemmed_content'] = self.raw_data['text'].apply(self.porter_stemming)
        print("Stemming operations completed...")
        return self.raw_data

    def __del__(self):
        pass