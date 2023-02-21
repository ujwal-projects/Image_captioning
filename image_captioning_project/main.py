import base64
import pickle
from fastapi import FastAPI,Request,Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from PIL import Image
import numpy as np
import io
from keras.models import Model
from keras.applications import ResNet50
import tensorflow as tf


cnn_lstm_model_30k = load_model('./models/cnn_lstm_30k/final_model.h5')
resnet_model = load_model('./models/cnn_lstm_30k/resnet50_model.h5')
with open('./models/cnn_lstm_30k/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
max_length = 74


cnn_lstm_model_8k = load_model('./models/cnn_lstm_8k/final_model.h5')
VGG_model = load_model('./models/cnn_lstm_8k/vgg19_model.h5')
with open('./models/cnn_lstm_8k/tokenizer.pickle', 'rb') as handle:
    tokenizer_8k = pickle.load(handle)
max_length_8k = 34


def predict_caption_8k(img, model, model_tokenizer, model_max_length):

    from tensorflow.keras.preprocessing import image
    x = image.img_to_array(img)
    #x = x.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    x = np.expand_dims(x,axis=0)
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
    x = preprocess_input(x)
    feature = VGG_model.predict(x, verbose=0)
    y_pred = predict_caption(model, feature, model_tokenizer, model_max_length)
    y_pred = " ".join(y_pred.split()[1:-1])
    return y_pred


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    image = image.reshape(1,-1)
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text


app = FastAPI(title='Image Captioning')
templates = Jinja2Templates(directory='templates/')


@app.get("/")
def root():
    return{'Project':'Image Captioning'}


@app.get("/captioning", tags=['HTML Connect'])
def form_data(request:Request):
    result = 'Please Submit Image'
    caption = None
    return templates.TemplateResponse('client_ui.html',context={'request':request,'result':result, 'cnn_lstm_30k_model_caption' : caption, 'cnn_lstm_8k_model_caption' : caption})


@app.post("/captioning",tags =['HTML Connect'])
def form_data(request:Request, img:bytes = File(...)):

    image_base64 = io.BytesIO(img)
    image_base64 = base64.b64encode(image_base64.getvalue()).decode("utf-8")

    pil_image = Image.open(io.BytesIO(img))
    img = np.asarray(pil_image.resize((224,224)))[...,:3]

    from tensorflow.keras.preprocessing import image
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    processed_img= tf.keras.applications.resnet50.preprocess_input(x, data_format=None)
    features_from_resnet = resnet_model.predict(processed_img).reshape(2048,)

    caption = predict_caption(cnn_lstm_model, features_from_resnet, tokenizer, max_length)
    caption = " ".join(caption.split()[1:-1])


    predicted_caption_8k = predict_caption_8k(img, cnn_lstm_model_8k, tokenizer_8k, max_length_8k)

    return templates.TemplateResponse('client_ui.html',context={'request':request,'img':image_base64, 'result' : "image uploaded", 'cnn_lstm_30k_model_caption' : caption, 'cnn_lstm_8k_model_caption' : predicted_caption_8k})
