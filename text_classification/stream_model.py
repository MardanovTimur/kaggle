import tensorflow as tf
from pymystem3 import Mystem
from utils import load_dumped, get_dataframe, clean_text
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from keras.preprocessing.sequence import pad_sequences
from lenta_training import MAX_SEQUENCE_LENGTH

sess = tf.Session()


class ModelMeta(type):
    __tokenizer_path: str = 'data/tokenizer_lenta.dump'
    __model_name = 'models/weights-08-1.09.hdf5'
    __dummies_path = 'data/dummies.dump'

    def __init__(self, name, bases, attrs):
        self.tokenizer = load_dumped(self.__tokenizer_path)
        self.labels = load_dumped(self.__dummies_path)

        set_session(sess)
        self.model = load_model(self.__model_name)
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        super().__init__(name, bases, attrs)


class Model(metaclass=ModelMeta):

    def predict(self, text):
        cleaned_texts = clean_text([text, ])

        X = self.tokenizer.texts_to_sequences([cleaned_texts])
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        with self.graph.as_default():
            set_session(sess)
            y = self.model.predict(X, steps=1)

        labels = []
        for result in y:
            #  places indices
            rargsort = result.argsort()[::-1][:5]
            for indice in rargsort:
                labels.append(f'{self.labels[indice]}: {result[indice]}')
        return labels
