import re
import numpy as np
import gensim
import pymorphy2
import pandas as pd
import pickle
from os.path import exists
from os import makedirs
from os import listdir
import sys
import argparse

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)


class TextGenerator:
    index_count_dict = {"ADJ": 0, "NOUN": 1, "OTH": 2, "VERB": 3, "empty": 4}

    def __init__(self):
        self.__book_handler = self.BookHandler()
        self.__word_tokenize = None
        self.__n_grams_model = None
        self.__tag_word_generator = None
        self.__punctuation_clf = None

    def fit(self, book_names=None):
        print("Start of training")
        self.__word_tokenize = self.WordTokenize()
        if book_names is None or len(book_names) == 0:
            self.BookHandler.process_stdin()
            book_names = ["user_text.txt"]
        print("Start training of the N-gram model")
        self.__n_grams_model = self.NGramsHandler(book_names)
        self.__n_grams_model.fit()
        print("Finish training of the N-gram model")
        print("Start training of tag generator")
        self.__tag_word_generator = self.TagWordGenerator(np.random.choice(book_names, 1)[0])
        self.__tag_word_generator.fit()
        print("Start training of punctuation generator")
        self.__punctuation_clf = self.PunctuationClassifier(np.random.choice(book_names, 1)[0])
        self.__punctuation_clf.fit()
        print("Finish training of punctuation generator")

    def generate(self, prefix=None, length=100, advance=False):
        if advance:
            return self.generate_advanced(prefix, length)
        return self.generate_simple(prefix, length)

    def generate_advanced(self, prefix, length):
        if prefix is None:
            sequence = self.__n_grams_model.search_first_sentence(3)
        else:
            sequence = ". " + " ".join(self.__book_handler.preprocess_book(prefix))

        sequence_vec = np.zeros(300)
        sequence_tag = [4] * 4
        count_dict = {"NOUN": 0, "ADJ": 0, "VERB": 0, "OTH": 0}
        position_at_sentence = 0

        for word in sequence.split(" ")[1:]:
            tag, vector = self.__word_tokenize(word, to_tag=True, to_vector=True)
            sequence_tag.append(self.index_count_dict[tag])
            sequence_vec += vector
            position_at_sentence += 1
            count_dict[tag] += 1

        text = prefix.lstrip(". ").capitalize()
        sequence_tag = sequence_tag[-4::]
        sequence = " ".join(sequence.split(" ")[-4::])
        is_capital = 0

        if not self.__n_grams_model.exists(sequence):
            print("There is no such prefix, try entering another one or autogeneration")
            exit(0)

        for i in range(length):
            next_tag = int(self.__tag_word_generator.predict(sequence_tag, count_dict, position_at_sentence)[0])
            word = self.__n_grams_model.predict(sequence, next_tag)
            tag, vector = self.__word_tokenize(word, to_tag=True, to_vector=True)
            if word == ".":
                punct_mark = self.__punctuation_clf.predict(sequence_vec)
                sequence_tag = [4] * 4
                count_dict = {"NOUN": 0, "ADJ": 0, "VERB": 0, "OTH": 0}
                sequence_vec = np.zeros(300)
                position_at_sentence = 0
                word = punct_mark
                is_capital = 2
            text += ("" if word in (".", "!", "?") else " ") + (word.capitalize() if is_capital else word)
            is_capital = max(0, is_capital - 1)
            sequence = sequence.split(" ")
            self.move_list(sequence, word)
            sequence = " ".join(sequence)
            self.move_list(sequence_tag, next_tag)
            sequence_vec += vector
        return text

    def generate_simple(self, prefix, length):
        if prefix is None:
            sequence = self.__n_grams_model.search_first_sentence(3)
        else:
            sequence = ". " + " ".join(self.__book_handler.preprocess_book(prefix))
        sequence_vec = np.zeros(300)
        for word in sequence.split(" ")[1:]:
            vector = self.__word_tokenize(word, to_vector=True)[1]
            sequence_vec += vector
        text = prefix.lstrip(". ").capitalize()
        sequence = " ".join(sequence.split(" ")[-4::])
        is_capital = 0
        if not self.__n_grams_model.exists(sequence):
            print("There is no such prefix, try entering another one or autogenerate")
            exit(0)
        for i in range(length):
            word = self.__n_grams_model.predict(sequence)
            vector = self.__word_tokenize(word, to_vector=True)[1]
            if word == ".":
                punct_mark = self.__punctuation_clf.predict(sequence_vec)
                sequence_vec = np.zeros(300)
                word = punct_mark
                is_capital = 2
            text += ("" if word in (".", "!", "?") else " ") + (word.capitalize() if is_capital else word)
            is_capital = max(0, is_capital - 1)
            sequence = sequence.split(" ")
            self.move_list(sequence, word)
            sequence = " ".join(sequence)
            sequence_vec += vector
        return text

    @staticmethod
    def move_list(sequence, new_element):
        sequence.pop(0)
        sequence.append(new_element)

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)

    class NGramsHandler:

        def __init__(self, book_names):
            self.__n_grams = [{}, {}, {}, {}, {}]
            self.book_names = book_names
            self.__word_tokenize = TextGenerator.WordTokenize()

        def fit(self):
            for book_name in self.book_names:
                book_handler = TextGenerator.BookHandler(book_name)
                self.memorize_n_grams(book_handler.parse_book())
                del book_handler

            assert all(list([all([f.count(" ") == j for f in i.keys()]) for j, i in
                             enumerate(self.__n_grams)])), "Keys for N-grams were built incorrectly."
            assert all(list([all([f.count(" ") == 0 for f in i.items()]) for j, i in
                             enumerate(self.__n_grams)])), "The values of the N-grams were constructed incorrectly."

            self.convert_n_grams_to_frequency()
            assert all(list([all([abs(1 - sum(v.values())) <= 0.01 for v in i.values()]) for i in
                             self.__n_grams])), "Conversion to frequency distributions failed."

        def predict(self, sequence, next_tag=None):
            if next_tag is not None:
                for _ in range(4):
                    for count_grams in range(4, 1, -1):
                        new_sequence = self.get_slice(sequence, count_grams)
                        if new_sequence not in self.__n_grams[count_grams]:
                            continue
                        word = np.random.choice(list(self.__n_grams[count_grams][new_sequence].keys()), 1,
                                                list(self.__n_grams[count_grams][new_sequence].values()))[0]
                        tag = self.__word_tokenize(word, to_tag=True)[0]
                        if TextGenerator.index_count_dict[tag] == next_tag and word != "None":
                            return word
            for count_grams in range(3, -1, -1):
                new_sequence = self.get_slice(sequence, count_grams)
                if new_sequence in self.__n_grams[count_grams]:
                    return np.random.choice(list(self.__n_grams[count_grams][new_sequence].keys()), 1,
                                            list(self.__n_grams[count_grams][new_sequence].values()))[0]

        def memorize_n_grams(self, split_text):
            sequences = [[], [], [], [], []]
            for word in split_text:
                for i in sequences:
                    i.append(word)
                for counter, i in enumerate(sequences):
                    if len(i) == counter + 2:
                        key = " ".join(i[:-1])
                        if key in self.__n_grams[counter]:
                            if word in self.__n_grams[counter][key]:
                                self.__n_grams[counter][key][word] += 1
                            else:
                                self.__n_grams[counter][key][word] = 1
                        else:
                            self.__n_grams[counter][key] = {word: 1}
                        i.pop(0)

        def convert_n_grams_to_frequency(self):
            for i_grams in self.__n_grams:
                for sequences, words_after_sequences in i_grams.items():
                    words_after_sequences_copy = words_after_sequences.copy()
                    for word, count_word in words_after_sequences.items():
                        words_after_sequences_copy[word] = count_word / sum(words_after_sequences.values())
                    i_grams[sequences] = words_after_sequences_copy

        @staticmethod
        def get_slice(string, n):
            return " ".join(string.split(" ")[(-1 - n):])

        @staticmethod
        def search_first_sentence(n_gram=None):
            return np.random.choice(list((filter(lambda x: x[0] == ".", n_gram))))

        def exists(self, word):
            for j, i in enumerate(self.__n_grams):
                if self.get_slice(word, j) in i:
                    return True
            return False

    class TagWordGenerator:

        PATH_DATA = "data/next_word_generator.csv"

        def __init__(self, book_name):
            self.model = None
            self.__word_tokenize = TextGenerator.WordTokenize()
            if not exists(self.PATH_DATA):
                print("Start of data generation for the tag generator")
                self.generate_data(book_name)
                print("Finish of data generation for the tag generator")

        def fit(self):
            data = pd.read_csv(self.PATH_DATA)
            data.drop("Unnamed: 0", axis=1, inplace=True)
            x_train, y_train = np.array(data)[:, :-1], np.array(data)[:, -1]
            self.model = RandomForestClassifier(n_estimators=30, max_depth=40, class_weight="balanced_subsample",
                                                max_samples=0.8, n_jobs=-1)
            self.model.fit(x_train, y_train)

        def predict(self, sequence, count_dict, position_at_sentence):
            row = self.convert_to_row(sequence, count_dict, position_at_sentence).reshape(1, -1)[:, :-1]
            return self.model.predict(row)

        def generate_data(self, book_name):
            data = []
            sequence = [4] * 4
            count_dict = {"NOUN": 0, "ADJ": 0, "VERB": 0, "OTH": 0}
            position_at_sentence = 0
            book_handler = TextGenerator.BookHandler(book_name)
            for word in book_handler.parse_book():
                tag = self.__word_tokenize(word, to_tag=True)[0]
                data.append(
                    self.convert_to_row(sequence, count_dict, position_at_sentence,
                                        TextGenerator.index_count_dict[tag]))
                if word == ".":
                    sequence = [4] * 4
                    count_dict = {"NOUN": 0, "ADJ": 0, "VERB": 0, "OTH": 0}
                    position_at_sentence = 0

                sequence.pop(0)
                sequence.append(TextGenerator.index_count_dict[tag])
                count_dict[tag] += 1
                position_at_sentence += 1

            data = pd.DataFrame(data).drop_duplicates().sample(4000)
            if not exists("/".join(self.PATH_DATA.split("/")[:-1])):
                makedirs("/".join(self.PATH_DATA.split("/")[:-1]))
            data.to_csv(self.PATH_DATA)
            del book_handler

        @staticmethod
        def convert_to_row(sequence, count_dict, position_at_sentence, tag=None):
            row = np.zeros(20)
            for j, i in enumerate(sequence):
                row[j * 5 + i] = 1
            row = np.concatenate(
                [row, np.array(list(count_dict.values())), np.array([position_at_sentence]), np.array(list([tag]))],
                axis=0)
            return row

    class PunctuationClassifier:

        PATH_DATA = "data/punctuation_classifier.csv"

        def __init__(self, book_name):
            self.model = None
            self.__word_tokenize = TextGenerator.WordTokenize()
            self.__punct_dict = {".": 0, "!": 1, "?": 2}
            if not exists(self.PATH_DATA):
                print("Start of data generation for the punctuation generator")
                self.generate_data(book_name)
                print("Finish of data generation for the punctuation generator")

        def fit(self):
            data = pd.read_csv(self.PATH_DATA)
            data.drop("Unnamed: 0", axis=1, inplace=True)
            x_train, y_train = np.array(data)[:, :-1], np.array(data)[:, -1]
            estimators = [
                ('rf',
                 RandomForestClassifier(n_estimators=40, max_depth=100, class_weight="balanced_subsample", n_jobs=-1)),
                ('log_reg', make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="saga"))),
                ('knn', KNeighborsClassifier(n_neighbors=16))
            ]
            self.model = StackingClassifier(estimators=estimators, cv=5)
            self.model.fit(x_train, y_train)

        def predict(self, x):
            res = int(self.model.predict(x.reshape(1, -1)))
            return list(self.__punct_dict.keys())[list(self.__punct_dict.values()).index(res)]

        def generate_data(self, book_name):
            sentence = np.zeros(300)
            data = []
            memorize = {}
            book_handler = TextGenerator.BookHandler(book_name)
            for i in book_handler.parse_book():
                if i in self.__punct_dict.keys():
                    data.append(self.convert_to_row(sentence, self.__punct_dict[i]))
                    sentence = np.zeros(300)
                    continue
                if i in ("", None):
                    continue
                if i in memorize:
                    vec = memorize[i]
                else:
                    vec = self.__word_tokenize(i, to_vector=True)[1]
                    memorize[i] = vec
                sentence += vec if not (vec is None) else np.zeros(300)

            data = pd.DataFrame(np.array(data, dtype=float)).drop_duplicates().sample(10000)
            if not exists("/".join(self.PATH_DATA.split("/")[:-1])):
                makedirs("/".join(self.PATH_DATA.split("/")[:-1]))
            data.to_csv(self.PATH_DATA)

        @staticmethod
        def convert_to_row(sentence, punct_mark):
            row = np.concatenate([sentence, np.array([punct_mark])])
            return row

    class BookHandler:
        def __init__(self, book_name=None):
            self.__book_name = book_name

        @staticmethod
        def process_stdin():
            print("Write your text and add a symbol at the end \\")
            text = []
            while True:
                char = sys.stdin.read(1)
                if not char:
                    break
                if char == '\\':
                    break 
                text.append(char)
            text = ''.join(text)
            with open("user_text.txt", "w") as f:
                f.write(text)

        def parse_book(self):
            with open(self.__book_name, "r", encoding="utf-8") as f:
                book = f.read()
            return self.preprocess_book(book)

        @staticmethod
        def preprocess_book(book):
            # Remove Page title
            book = re.sub(r"(Глава|Страница)[А-Яа-яA-Za-z\s|.,;\-\d#]*\n+", " ", book)
            book = re.sub(r"[\d]+\n+", " ", book)
            # Removing HTML tags
            book = re.sub(r"<.*?>", '', book)
            # Removing metatags
            book = re.sub(r"(\n|\a|\f|\r|\t|\v|\\)+", ' ', book)
            # Changing quotation marks
            book = re.sub(r"([“”«»])", "\"", book)
            book = re.sub(r"’", "'", book)
            book = re.sub(r"([—–])", "-", book)
            # Removing extra punctuation marks
            book = re.sub(r"([,:\-…/#;*])", "", book)
            # Direct speech processing
            book = re.sub(r"\"([А-Яа-яA-Za-z0-9\s.!?']*)\"", "\g<1>", book)
            # Solving the problem with punctuation marks and extra spaces
            book = re.sub(r"\s{2,}", " ", book)
            book = re.sub(r"(\w)([.!?;])", "\g<1> \g<2>", book)
            return book.lower().split(" ")

    class WordTokenize:

        PATH_MODEL = "data/model.bin"

        POS_MAP_SIMPLE = {
            "ADJF": "ADJ",
            "ADJS": "ADJ",
            "ADVB": "OTH",
            "COMP": "OTH",
            "GRND": "VERB",
            "INFN": "VERB",
            "PRED": "OTH",
            "PRTF": "ADJ",
            "PRTS": "VERB",
            "NOUN": "NOUN",
            "NUMR": "OTH",
            "NPRO": "OTH",
            "PREP": "OTH",
            "CONJ": "OTH",
            "PRCL": "OTH",
            "INTJ": "OTH",
            "ADJ": "ADJ",
            "ADV": "OTH",
            "VERB": "VERB"
        }

        POS_MAP_ADVANCED = {
            "ADJF": "ADJ",
            "ADJS": "ADJ",
            "ADVB": "ADV",
            "COMP": "ADV",
            "GRND": "VERB",
            "INFN": "VERB",
            "PRED": "ADV",
            "PRTF": "ADJ",
            "PRTS": "VERB"
        }

        def __init__(self):
            if not exists(self.PATH_MODEL):
                raise FileNotFoundError(
                    f"The dictionary was not found. Please download it from the following link: "
                    f"http://vectors.nlpl.eu/repository/20/180.zip and put it in {self.PATH_MODEL}")

            self.morph = pymorphy2.MorphAnalyzer()
            self.gensim_model = gensim.models.KeyedVectors.load_word2vec_format(self.PATH_MODEL, binary=True)
            self.gensim_model.fill_norms()

        def __call__(self, word, to_tag=False, to_vector=False):
            tag = self.__get_tag(word) if to_tag else None
            vector = self.__get_vector(word) if to_vector else None
            return [tag, vector]

        def __get_tag(self, word):
            tag = str(self.morph.parse(word)[0].tag.POS)
            if tag == "None" or word == "":
                tag = "OTH"
            else:
                tag = self.POS_MAP_SIMPLE[tag]
            return tag

        def __get_vector(self, word):
            parse_result = self.morph.parse(word)
            if len(parse_result) == 0:
                return np.zeros(300)
            parse_result = parse_result[0]
            pos = parse_result.tag.POS
            if pos is None:
                return np.zeros(300)
            pos = self.POS_MAP_ADVANCED.get(pos, pos)
            lemma = parse_result.normal_form
            for token in [word + "_" + pos, lemma + "_" + pos, word + "_" + "PROPN", lemma + "_" + "PROPN"]:
                index = self.gensim_model.key_to_index.get(token)
                if index is not None:
                    return self.gensim_model.get_vector(index)
            return np.zeros(300)


def preprocess_arg(arg):
    arg = arg + "/" if arg[-1] != "/" else ""
    if not exists(arg):
        raise FileNotFoundError(f"{arg} does not exist!")
    return arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A text generator based on an N gram model')
    parser.add_argument('-input', '--input-dir', type=str, default='data/', help='Input dir for texts')
    parser.add_argument('-m', '--model', type=str, default='data/model.pickle', help='Output dir for model')
    args = parser.parse_args()
    input_dir = preprocess_arg(args.input_dir)
    model_path = args.model
    if re.match(r"[A-Za-zА-Яа-я0-9.\s/\\]+.pickle$", model_path) is None:
        print(f"Вы ввели некорректный путь до файла модели, поэтому он сброшен до дефолтного: data/model.pickle")
        model_path = "data/model.pickle"

    books = list([input_dir + i for i in filter(lambda x: re.match(r"[A-Za-zА-Яа-я0-9.\s]+.txt$", x) is not None,
                                                listdir(input_dir))])
    model = TextGenerator()
    model.fit(books)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
