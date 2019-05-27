import urllib.request
import os
import tarfile

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)

if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open(filepath, 'r:gz')
    result = tfile.extractall('data/')


from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
    path = "data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path+f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts


y_train, train_text = read_files('train')
y_test, test_text = read_files('test')

token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN


model = Sequential()
model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(SimpleRNN(units=16))
# model.add(LSTM(32))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history = model.fit(x=x_train, y=y_train,
                          validation_split=0.2, epochs=10, batch_size=100, verbose=2)

scores = model.evaluate(x=x_test, y=y_test)
print(scores[1])


def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen=100)
    predict_result = model.predict_classes(pad_input_seq)
    print(predict_result[0][0])


input_text = '''Was lifeless, characters had no emotion or personality , 
                found myself rooting for gaston, he was the only cool one. Modern autotuned garbage. 
                CGI was awful and way overdone. Nothing like the original. Songs were long and annoying. 
                Dialogue felt rushed like they tried to cram all the stuff from the original into it but 
                ran out of time.'''

predict_review(input_text)

