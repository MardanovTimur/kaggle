import pathlib
import pickle
import numpy as np

from keras.models import load_model as model_loader

from utils import preprocess, create_dataset

MODEL_DIR = pathlib.Path(__file__).parent / 'models'

MODEL_NAME = 'weights-06-6.66.hdf5'

TEXT = """
Наталья Ильинишна очень хорошо со мной обходится,
    — сказал Борис.
    — Я не могу жаловаться, — сказал он.
— Оставьте, Борис, вы такой дипломат
(слово дипломат было в большом ходу у детей в том особом значении, какое они придавали этому слову); даже скучно, — сказала Наташа оскорбленным, дрожащим голосом.
— За что она ко мне пристает?
— Ты этого никогда не поймешь,
— сказала она, обращаясь к Вере, — потому что ты никогда никого не любила; у тебя сердца нет, ты только madame de Genlis (это прозвище, считавшееся очень обидным, было дано Вере Николаем),
и твое первое удовольствие — делать неприятности другим. Ты кокетничай с Бергом сколько хочешь, — проговорила она скоро.
"""


def load_model(model_name=MODEL_NAME):
    return model_loader(str(MODEL_DIR / model_name))


def load_tokenizer(dump_name='models/tokenizer.dump'):
    with open(dump_name, 'rb') as file:
        return pickle.load(file)


def load_hot_encodings(dump_name='models/labels.dump'):
    with open(dump_name, 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    tokenizer = load_tokenizer()
    labels = load_hot_encodings()

    preprocessed_text = preprocess(TEXT, read=True)
    X = tokenizer.texts_to_sequences(preprocessed_text)

    X_test, y = create_dataset(np.array(X), 4)

    print('Preprocessed_text:', preprocessed_text)

    model = load_model()
    results = model.predict(X_test)
    print(results)

    for result in results:
        rargsort = result.argsort()[::-1][:5]
        for indice in rargsort:
            next_word = tokenizer.index_word[labels[indice]]
            print(f'{next_word}: {result[indice]}')
        print('\n' + '-' * 10)

