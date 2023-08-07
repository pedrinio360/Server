
"""Server
"""
import os
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from flask import Flask, jsonify, request

def prepararModelo():
    global model_kichwa_spanish
    global model_spanish_kichwa
    global len_sentences
    global num_vocab
    global texto_source_spanish_vectorizado
    global texto_target_kichwa_vectorizado
    global texto_source_kichwa_vectorizado
    global texto_target_spanish_vectorizado
    global busqueda_kichwa
    global busqueda_spanish

    sentences_spanish_kichwa = load_data('data/pares_spanish_kichwa.txt')
    sentences_kichwa_spanish = load_data('data/pares_kichwa_spanish.txt')

    len_sentences = 7
    num_vocab = 5000

    model_spanish_kichwa = tf.saved_model.load('data/model_spanish_kichwa')
    model_kichwa_spanish = tf.saved_model.load('data/model_kichwa_spanish')

    print('Modelo preparado')

    pares_spanish_kichwa = get_pairs(sentences_spanish_kichwa)
    pares_kichwa_spanish = get_pairs(sentences_kichwa_spanish)

    texto_source_vectorizado_sk, texto_target_vectorizado_sk = vectorizar_spanish_kichwa()
    texto_source_spanish_vectorizado, texto_target_kichwa_vectorizado, busqueda_kichwa = preprocess_dataset(
        pares_spanish_kichwa, 
        texto_source_vectorizado_sk, 
        texto_target_vectorizado_sk)
    
    texto_source_vectorizado_ks, texto_target_vectorizado_ks = vectorizar_kichwa_spanish()
    texto_source_kichwa_vectorizado, texto_target_spanish_vectorizado, busqueda_spanish = preprocess_dataset(
        pares_kichwa_spanish, 
        texto_source_vectorizado_ks, 
        texto_target_vectorizado_ks)

    print('Procesamiento de datos listo')

def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf8') as f:
        data = f.read()
    return data.split('\n')

def get_pairs(lines):
    pairs = []
    for line in lines:
        item1, item2 = line.split('[start]')
        pairs.append((item1, '[start]' + item2))
    return pairs

def get_standardization(text):
    modificar_chars = string.punctuation
    modificar_chars = modificar_chars.replace('[', '')
    modificar_chars = modificar_chars.replace(']', '')

    texto = tf.strings.lower(text)
    return tf.strings.regex_replace(
        texto, f'[{re.escape(modificar_chars)}]', '')

def vectorizar_spanish_kichwa():
    texto_source_vectorizado = layers.experimental.preprocessing.TextVectorization(
        output_mode = "int",
        output_sequence_length = len_sentences,
        max_tokens = num_vocab,
        standardize = get_standardization
    )
    texto_target_vectorizado = layers.experimental.preprocessing.TextVectorization(
        output_mode = "int",
        output_sequence_length = len_sentences + 1,
        max_tokens = num_vocab,
        standardize = get_standardization
    )

    return texto_source_vectorizado, texto_target_vectorizado

def vectorizar_kichwa_spanish():
    texto_source_vectorizado = layers.experimental.preprocessing.TextVectorization(
        output_mode = "int",
        output_sequence_length = len_sentences,
        max_tokens = num_vocab
    )
    texto_target_vectorizado = layers.experimental.preprocessing.TextVectorization(
        output_mode = "int",
        output_sequence_length = len_sentences + 1,
        max_tokens = num_vocab,
        standardize = get_standardization
    )

    return texto_source_vectorizado, texto_target_vectorizado

def preprocess_dataset(pares, texto_source, texto_target):
    textos_train_source = [par[0] for par in pares]
    textos_train_target = [par[1] for par in pares]
    texto_source.adapt(textos_train_source)
    texto_target.adapt(textos_train_target)

    vocab = texto_target.get_vocabulary()
    busqueda = dict(zip(range(len(vocab)), vocab))
    return texto_source, texto_target, busqueda


def get_prediction(input_sentence, busqueda, texto_source_vectorizado, texto_target_vectorizado, isKichwaToSpanish):
    tokenized_input_sentence = texto_source_vectorizado([input_sentence])
    sentence_decoded = "[start]"

    for i in range(len_sentences):
        sentence_target_tokenized = texto_target_vectorizado(
            [sentence_decoded])[:, :-1]
        if isKichwaToSpanish :
            predicciones = model_kichwa_spanish(
            [tokenized_input_sentence, sentence_target_tokenized])
        else :
            predicciones = model_spanish_kichwa(
            [tokenized_input_sentence, sentence_target_tokenized])
        index_token = np.argmax(predicciones[0, i, :])
        token = busqueda[index_token]
        sentence_decoded += " " + token
        if token == "[end]":
            break
    return sentence_decoded

def traducir(input_sentence, isKichwaToSpanish):
    traduccion = ''
    prediccion = ''
    if isKichwaToSpanish:
        prediccion = get_prediction(input_sentence, 
                                    busqueda_spanish, 
                                    texto_source_kichwa_vectorizado, 
                                    texto_target_spanish_vectorizado, 
                                    True)
    else:
        prediccion = get_prediction(input_sentence, 
                                    busqueda_kichwa, 
                                    texto_source_spanish_vectorizado, 
                                    texto_target_kichwa_vectorizado, 
                                    False)        
    
    traduccion = prediccion.replace("[start]", "").replace("[end]", "").strip()
    return traduccion

def create_app():
    prepararModelo()
    return Flask(__name__)

app = create_app()

#flask
@app.route('/')
def get_server_status():
    return 'SERVER UP'


@app.route('/kichwa-espanol/', methods=['POST'])
def traducir_kichwa_espanol():
    """API request
    """
    texto = request.json['texto']
    if texto:
        response = traducir(texto, True)
        print("Enviar respuesta")
        print(response)
        print("Fin de Peticion")
    
        return response
    else:
        response = jsonify({
            'message': 'Texto a traducir no encontrado' + request.url,
            'status': '404'
        })
        response.status_code = 404
        return response

@app.route('/espanol-kichwa/', methods=['POST'])
def traducir_espanol_kichwa():
    """API request
    """
    texto = request.json['texto']
    if texto:
        response = traducir(texto, False)
        print("Enviar respuesta")
        print(response)
        print("Fin de Peticion")
    
        return response
    else:
        response = jsonify({
            'message': 'Texto a traducir no encontrado' + request.url,
            'status': '404'
        })
        response.status_code = 404
        return response

if __name__ == "__main__":
    app.run(debug=False)
