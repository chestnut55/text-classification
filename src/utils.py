from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data():
    print('(1) load texts...')

    train_texts = open('../data/train_contents.txt').read().split('\n')
    train_labels = open('../data/train_labels.txt').read().split('\n')
    test_texts = open('../data/test_contents.txt').read().split('\n')
    test_labels = open('../data/test_labels.txt').read().split('\n')

    return train_texts, train_labels, test_texts, test_labels


def tf_idf(train_texts, test_texts):
    print('(2) doc to var...')

    count_v0 = CountVectorizer()
    all_text = train_texts + test_texts
    count_v0.fit_transform(all_text)
    count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_train = count_v1.fit_transform(train_texts)
    print("the shape of train is " + repr(counts_train.shape))
    count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_test = count_v2.fit_transform(test_texts)
    print("the shape of test is " + repr(counts_test.shape))

    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit(counts_train).transform(counts_train)
    test_data = tfidftransformer.fit(counts_test).transform(counts_test)

    return train_data, test_data
