from keras.models import model_from_json
from keras.models import Model, Sequential


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print loaded_model.summary()

decoder = Model(inputs=loaded_model.get_layer('code').input,
                outputs=loaded_model.output)
print decoder.summary()
