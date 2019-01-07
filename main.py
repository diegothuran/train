from keras.models import load_model
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def load_tokenizer():

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


model = load_model('keras.h5')

tokenizer_obj = load_tokenizer()

max_length = 339

test_sample1 = [[x] for x in "A minha cadela adorou a ração".split()]
test_sample2 = "A cadela da atendente é ruim"
test_samples = [test_sample2]

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

print(model.predict(x=test_samples_tokens_pad))