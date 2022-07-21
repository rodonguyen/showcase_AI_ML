import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
import re
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from time import process_time
import pickle
import operator
import tensorflow

# Load the data


def load_data(path):
    # Load data
    full_data = pd.read_csv(path, header=0)

    # # Set as categorical
    # full_data["is_retweet"] = pd.Categorical(full_data.is_retweet)
    # full_data["Type"] = pd.Categorical(full_data.Type)
    full_data.iloc[:, 5] = pd.Categorical(full_data.iloc[:, 5])
    full_data.iloc[:, 7] = pd.Categorical(full_data.iloc[:, 7])

    x_train, y_train = full_data.iloc[:, 1:-1], full_data.iloc[:, -1]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=3000, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42)

    print('\nx shape   y shape')
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    # reset indices
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return x_train, y_train, x_val, y_val, x_test, y_test


def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)


def lower_and_seperate(x):
    copy = x.copy()
    for number, t in enumerate(copy):
        t = re.sub(r"[^a-z0-9#@\*'\":\-\n%,\.;?!]+", " ", str(t).lower())
        t = re.sub(r"#", " # ", t)
        t = re.sub(r"@", " @ ", t)
        t = re.sub(r"\*", " * ", t)
        t = re.sub(r"\'", " ' ", t)
        t = re.sub(r"\"", " \" ", t)
        t = re.sub(r"\:", " : ", t)
        t = re.sub(r"\-", " - ", t)
        t = re.sub(r"\%", " % ", t)
        t = re.sub(r"\,", " , ", t)
        t = re.sub(r"\.", " . ", t)
        t = re.sub(r"\;", " ; ", t)
        t = re.sub(r"\?", " ? ", t)
        t = re.sub(r"\!", " ! ", t)
        copy.iloc[number] = t
    return copy


def tokenise(x_train, x_val, x_test, char_level=False):
    x_train_las = lower_and_seperate(x_train)
    x_val_las = lower_and_seperate(x_val)
    x_test_las = lower_and_seperate(x_test)

    tokenizer = Tokenizer(num_words=50000,
                          filters='$&()+/<=>[\\]^_`{|}~\t',
                          char_level=char_level)
    tokenizer.fit_on_texts(x_train_las)

    train_sequences = []
    for seq in tokenizer.texts_to_sequences_generator(x_train_las):
        train_sequences.append(seq)

    val_sequences = []
    for seq in tokenizer.texts_to_sequences_generator(x_val_las):
        val_sequences.append(seq)

    test_sequences = []
    for seq in tokenizer.texts_to_sequences_generator(x_test_las):
        test_sequences.append(seq)

    max_length = max(find_max_list(train_sequences),
                     find_max_list(val_sequences),
                     find_max_list(test_sequences))

    x_train_tokenised = np.array(pad_sequences(
        train_sequences, maxlen=max_length, padding='post'))
    x_val_tokenised = np.array(pad_sequences(
        val_sequences, maxlen=max_length, padding='post'))
    x_test_tokenised = np.array(pad_sequences(
        test_sequences, maxlen=max_length, padding='post'))

    return x_train_tokenised, x_val_tokenised, x_test_tokenised, \
        x_train_las, x_val_las, x_test_las, max_length, tokenizer


def eval_model(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_train, Y_train, normalize=None, ax=ax)
    conf.ax_.set_title('Training Set Performance')
    ax = fig.add_subplot(1, 2, 2)
    conf = ConfusionMatrixDisplay.from_estimator(
        model, X_test, Y_test, normalize=None, ax=ax)
    conf.ax_.set_title('Test Set Performance')
    start_time = process_time()
    pred = model.predict(X_test)
    end_time = process_time()
    time = end_time - start_time
    print("Inference time for test set is {:0.3f} s".format(time))
    print(classification_report(Y_test, pred))


def impute_dataframe(df, columns, missing_val=np.nan, strategy='mean', verbose=False):
    """
    Imputes columns based on some missing value and strategy

    Author:
        @instantpants : Thomas Fabian

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to impute

    columns : list[str]
        Columns to impute

    missing_val : type, default=np.nan
        Missing value type to impute

    strategy : str, default='mean'
        The imputation strategy.
        - If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
        - If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
        - If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
        - If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.

    verbose : bool
      Prints out debug values if true

    Returns
    =======
    df_imp : pd.DataFrame
        Dataframe after imputation

    imputer : sklearn.SimpleImputer
        Imputer object fit on supplied columns (See appendix 1.0)

    APPENDIX
    ========
    1.0 SimpleImputer
      https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

      Imputer Attributes
      imputer.statistics_ :
        The imputation fill value for each feature. Computing statistics can result in np.nan values. 
        During transform, features corresponding to np.nan statistics will be discarded.

      imputer.indicator_ :
        Indicator used to add binary indicators for missing values. None if add_indicator=False.

      imputer.n_features_in :
        Number of features seen during fit.

      imputer.feature_names_in_
        Names of features seen during fit. Defined only when X has feature names that are all strings.
    """
    imputer = SimpleImputer(missing_values=missing_val, strategy=strategy)
    imputer.fit(df[columns])
    fea = imputer.feature_names_in_
    # fead = df.columns.difference(fea)

    transform = imputer.transform(df[fea])
    transform = pd.DataFrame(transform, columns=fea, index=df.index)
    df_imp = df.copy()
    # df_imp = pd.DataFrame(transform, columns=fea, index=df.index)
    df_imp[fea] = transform[fea]

    if verbose:
        print(df.shape, df_imp.shape)
        print("NAN Counts:\n", df_imp.isna().sum(axis=0))

    return df_imp, imputer

# standardise all datasets (for specific columns) by the training set's mean and standard deviation


def standardise(train_x, val_x, test_x, columns):
    mu = np.mean(train_x[columns])
    sigma = np.std(train_x[columns])
    train_x_std = train_x.copy()
    val_x_std = val_x.copy()
    test_x_std = test_x.copy()
    train_x_std[columns] = (train_x[columns] - mu) / sigma
    val_x_std[columns] = (val_x[columns] - mu) / sigma
    test_x_std[columns] = (test_x[columns] - mu) / sigma

    return train_x_std, val_x_std, test_x_std


def savedata(data, filename):
    with open(f"{filename}", "wb") as f:
        pickle.dump(data, f)


def loaddata(filename):
    with open(f"{filename}", "rb") as f:
        return pickle.load(f)


def num_matching_words(dataset, indices, column, pattern, vocab, verbose=False, spam_count_dict=None, quality_count_dict=None):
    '''
    Return a dictionary of indices and how many words in the document were found in a given vocabulary.

    Parameters
    ==========
    dataset: pd.DataFrame
        A dataset with a text/document column.

    indices: iterable
        An iterable object of indices from the dataset.

    column: str
        The name of the text column.

    pattern: str
        The regexp pattern defining words.

    vocab: iterable
        An iterable object containing the vocabulary of words to compare the document's words to.

    verbose: bool
        Whether to print the document and found words to the console while processing.

    Returns
    =======
    num_in_vocab: dict
        A dictionary of indices and how many words in the document were found in a given vocabulary.
    '''
    num_in_vocab = dict()
    for idx in indices:
        if verbose:
            print("Document contents:")
            print(dataset.iloc[idx, :][column])
            print()
        regex = re.compile(pattern)
        words = regex.findall(dataset.iloc[idx, :][column].lower())
        if verbose:
            print("Pattern found the following words:")
            print(words)
            print()
        unique_words = {word for word in words if word in vocab}
        print(unique_words)
        num_in_vocab[idx] = len(unique_words)
        if verbose:
            print(
                f"{num_in_vocab[idx]} words from the document were found in the vocabulary.")
            print()
            if spam_count_dict and quality_count_dict:
                word_count = dict()
                for word in words:
                    if word in vocab:
                        if word in spam_count_dict:
                            word_spam_count = spam_count_dict[word]
                        else:
                            word_spam_count = None
                        if word in quality_count_dict:
                            word_quality_count = quality_count_dict[word]
                        else:
                            word_quality_count = None
                        word_count[word] = (
                            word_spam_count, word_quality_count)
                print(
                    "The following words were in the vocabulary: {}".format(word_count))
    return num_in_vocab


def num_matching_words_keras(dataset, indices, pattern, vocab, column=None, verbose=False, spam_count_dict=None, quality_count_dict=None):
    '''
    Return a dictionary of indices and how many words in the document were found in a given vocabulary. Works with the Keras Tokenizer.

    Parameters
    ==========
    dataset: pd.DataFrame
        A dataset with a text/document column.

    indices: iterable
        An iterable object of indices from the dataset.

    column: str
        The name of the text column.

    pattern: str
        The regexp pattern defining words.

    vocab: iterable
        An iterable object containing the vocabulary of words to compare the document's words to.

    verbose: bool
        Whether to print the document and found words to the console while processing.

    Returns
    =======
    num_in_vocab: dict
        A dictionary of indices and how many words in the document were found in a given vocabulary.
    '''
    num_in_vocab = dict()
    for idx in indices:
        if verbose:
            print("Document contents:")
            if column:
                print(dataset.iloc[idx, :][column])
            else:
                print(dataset.iloc[idx])
            print()
        regex = re.compile(pattern)
        if column:
            words = regex.findall(dataset.iloc[idx, :][column].lower())
        else:
            words = regex.findall(dataset.iloc[idx].lower())
        if verbose:
            print("Pattern found the following words:")
            print(words)
            print()
        unique_words = {word for word in words if word in vocab}
        print(unique_words)
        num_in_vocab[idx] = len(unique_words)
        if verbose:
            print(
                f"{num_in_vocab[idx]} words from the document were found in the vocabulary.")
            print()
            if spam_count_dict and quality_count_dict:
                word_count = dict()
                for word in words:
                    if word in vocab:
                        if word in spam_count_dict:
                            word_spam_count = spam_count_dict[word]
                        else:
                            word_spam_count = None
                        if word in quality_count_dict:
                            word_quality_count = quality_count_dict[word]
                        else:
                            word_quality_count = None
                        word_count[word] = (
                            word_spam_count, word_quality_count)
                print(
                    "The following words were in the vocabulary: {}".format(word_count))
    return num_in_vocab

# Pull out words we found, check their freqs
# def find_words_cnt(transformer, x, y):
#     '''
#     Return dictionaries of words and their pure word count.

#     Parameters
#     ==========
#     transformer : sklearn.compose.ColumnTransformer
#         A TF-IDF transformer as per the BoW-SVM file.
#     x : pd.DataFrame
#         The features
#     y : pd.DataFrame
#         The response "Spam" or "Quality"

#     Returns
#     =======
#     word_tfidf: dict
#         A dictionary containing all words in the corpus and their pure word count
#     spam_words: dict
#         A dict containing all words in the corpus and their pure word count from all spam tweets (as per y)
#     quality_words:
#         A dict containing all words in the corpus and their pure word count from all quality tweets (as per y)
#     '''
#     # Used to pull out words from TF-IDF, as TF-IDF outputs are sparse matrices that have no column names.
#     # If you have a different way to get your words, this is the line you want to replace with your way.
#     words = [word.replace("tweet__", "").replace("location__", "") for word in transformer.get_feature_names_out()][:-4]

#     # Get counts
#     tfidf_count = x.astype('bool').sum(axis=0)
#     word_tfidf = dict()
#     spam_words = dict()
#     quality_words = dict()

#     # Get all counts
#     for idx, word in enumerate(words):
#         word_tfidf[word] = tfidf_count[0, idx]

#     # Get specific counts
#     spam_tfidf_count = x[y=="Spam"][:,:-4].astype('bool').sum(axis=0)
#     quality_tfidf_count = x[y=="Quality"][:,:-4].astype('bool').sum(axis=0)
#     for idx, word in enumerate(words):
#         spam_words[word] = spam_tfidf_count[0, idx]
#     for idx, word in enumerate(words):
#         quality_words[word] = quality_tfidf_count[0, idx]

#     return word_tfidf, spam_words, quality_words

# Pull out words we found, check their freqs


def find_words_sum(transformer, x, y, only_tweets=False):
    '''
    Return dictionaries of words and their total TF-IDF values.

    Parameters
    ==========
    transformer : sklearn.compose.ColumnTransformer
        A TF-IDF transformer as per the BoW-SVM file.
    x : pd.DataFrame
        The features
    y : pd.DataFrame
        The response "Spam" or "Quality"

    Returns
    =======
    word_tfidf: dict
        A dictionary containing all words in the corpus and their total TF-IDF values
    spam_words: dict
        A dict containing all words in the corpus and their total TF-IDF values from all spam tweets (as per y)
    quality_words:
        A dict containing all words in the corpus and their total TF-IDF values from all quality tweets (as per y)
    '''
    # Used to pull out words from TF-IDF, as TF-IDF outputs are sparse matrices that have no column names.
    # If you have a different way to get your words, this is the line you want to replace with your way.
    if only_tweets:
        words = [word.replace("tweet__", "") for word in transformer.get_feature_names_out(
        ) if word.startswith("tweet__")][:-4]
    else:
        words = [word.replace("tweet__", "").replace("location__", "")
                 for word in transformer.get_feature_names_out()][:-4]

    # Grab the sum
    tfidf_count = x.sum(axis=0)

    # Set up dicts
    word_tfidf = dict()
    spam_words = dict()
    quality_words = dict()

    # Get all frequencies
    for idx, word in enumerate(words):
        word_tfidf[word] = tfidf_count[0, idx]

    # Get the spam/quality-specific ones
    spam_tfidf_count = x[y == "Spam"][:, :-4].sum(axis=0)
    quality_tfidf_count = x[y == "Quality"][:, :-4].sum(axis=0)
    for idx, word in enumerate(words):
        spam_words[word] = spam_tfidf_count[0, idx]
    for idx, word in enumerate(words):
        quality_words[word] = quality_tfidf_count[0, idx]

    return word_tfidf, spam_words, quality_words

# Pull out words we found, check their freqs
# def find_words_mean(transformer, x, y):
#     '''
#     Return dictionaries of words and their mean TF-IDF values.

#     Parameters
#     ==========
#     transformer : sklearn.compose.ColumnTransformer
#         A TF-IDF transformer as per the BoW-SVM file.
#     x : pd.DataFrame
#         The features
#     y : pd.DataFrame
#         The response "Spam" or "Quality"

#     Returns
#     =======
#     word_tfidf: dict
#         A dictionary containing all words in the corpus and their mean TF-IDF values
#     spam_words: dict
#         A dict containing all words in the corpus and their mean TF-IDF values from all spam tweets (as per y)
#     quality_words:
#         A dict containing all words in the corpus and their mean TF-IDF values from all quality tweets (as per y)
#     '''
#     # Used to pull out words from TF-IDF, as TF-IDF outputs are sparse matrices that have no column names.
#     # If you have a different way to get your words, this is the line you want to replace with your way.
#     words = [word.replace("tweet__", "") for word in transformer.get_feature_names_out() if word.startswith("tweet__")][:-4]

#     # Get means
#     tfidf_count = x.mean(axis=0)

#     # Set dicts
#     word_tfidf = dict()
#     spam_words = dict()
#     quality_words = dict()

#     # Get all means in a dict
#     for idx, word in enumerate(words):
#         word_tfidf[word] = tfidf_count[0,idx]

#     # Get specific dicts
#     spam_tfidf_count = x[y=="Spam"][:,:-4].mean(axis=0)
#     quality_tfidf_count = x[y=="Quality"][:,:-4].mean(axis=0)
#     for idx, word in enumerate(words):
#         spam_words[word] = spam_tfidf_count[0,idx]
#     for idx, word in enumerate(words):
#             quality_words[word] = quality_tfidf_count[0,idx]

#     return word_tfidf, spam_words, quality_words


def sort_spam_quality(spam_words, quality_words):
    '''
    Return a sorted list of spam and quality words in ascending order as a list of tuples.

    Parameters
    ==========
    spam_words: dict
        A dictionary from 

    Returns
    =======
    num_in_vocab: dict
        A dictionary of indices and how many words in the document were found in a given vocabulary.
    '''
    sorted_quality_words = sorted(
        quality_words.items(), key=operator.itemgetter(1))
    sorted_spam_words = sorted(spam_words.items(), key=operator.itemgetter(1))
    return sorted_quality_words, sorted_spam_words


def stats(counts):
    print(f"Count: {np.count_nonzero(counts)}")
    print(f"Mean: {np.mean(counts)}")
    print(f"Median: {np.median(counts)}")
    print(f"Variance: {np.var(counts)}")
    print(f"Standard deviation: {np.std(counts)}")
    print(f"Max: {np.max(counts)}")
    print(f"Min: {np.min(counts)}")
    print()


def normalise_histogram(hist):
    return hist / np.sum(hist)


def histogram_intersection(hist1, hist2):
    total = 0
    for i, j in zip(hist1, hist2):
        total += min(i, j)
    return total


def transform_to_float(y_train, y_val, y_test):
    y_train_int = (y_train == 'Spam') * 1
    y_val_int = (y_val == 'Spam') * 1
    y_test_int = (y_test == 'Spam') * 1

    y_train_int = tensorflow.cast(y_train_int, tensorflow.float32)
    y_val_int = tensorflow.cast(y_val_int, tensorflow.float32)
    y_test_int = tensorflow.cast(y_test_int, tensorflow.float32)

    return y_train_int, y_val_int, y_test_int
