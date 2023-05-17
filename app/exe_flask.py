from flask import Flask, request, Response
from keras.optimizers import Adam
from flask_cors import CORS

import pickle
import tensorflow as tf

application = Flask(__name__)
CORS(application)

with open('vectorizador_coches.pck', 'rb') as f:
  dv = pickle.load(f)
model = tf.keras.models.load_model('modelo_coches.hdf5')
# Optimizador
optimizer = Adam(learning_rate=0.01)

# Compilar el modelo 
model.compile(loss='mean_squared_error', optimizer=optimizer)

@application.route('/flask', methods=['GET'])
def flask():
    return '<p>https://flask.palletsprojects.com/en/2.2.x/</p>'

@application.post('/api/coche')
def prediccion():
    coche = request.get_json()
    for key, value in coche.items():
        if key in ('km', 'cubicCapacity', 'hp', 'doors', 'year'):
            coche[key] = float(value) 

    if coche is not None:
        print('data:',[dict(coche)])
        X_train = dv.transform([dict(coche)])
        precio = model.predict(X_train)
        
        return {'precio': round(float(precio[0]), 2)}
    else:
        return 'Error'
    

application.run()