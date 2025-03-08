import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from io import BytesIO
from decouple import config
from google.cloud import storage
from tensorflow.keras.models import load_model
import os
import warnings
warnings.filterwarnings(
    "ignore", message="Skipping variable loading for optimizer")

# Cargar datos de los animales y los usuarios desde Firebase
ANIMAL_KEY = config('ANIMAL_KEY')
USER_KEY = config('USER_KEY')

# Cargar la clave de la cuenta de servicio
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "api-matchpet-7f1d8f4c1fd7.json"

# Ensure this is above the load_model_from_gcs function


@register_keras_serializable()
def compute_similarity(vectors):
    user_vec, animal_vec = vectors
    return K.exp(-K.sum(K.abs(user_vec - animal_vec), axis=1, keepdims=True))


# Cargar el modelo desde Google Cloud Storage


def load_model_from_gcs(bucket_name, model_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(model_name)
    model_path = "/tmp/model.keras"  # Ruta temporal en el servidor

    # Descargar el modelo a la ruta temporal
    blob.download_to_filename(model_path)

    # Cargar el modelo
    model = load_model(model_path, custom_objects={
                       'compute_similarity': compute_similarity})
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Usa esta función para cargar el modelo en el inicio del servidor
model = load_model_from_gcs("matchpeet_ai", "model.keras")


def load_data_from_firebase(user_id):
    response_animal = requests.get(ANIMAL_KEY)
    response_user = requests.get(USER_KEY)

    if response_animal.status_code == 200 and response_user.status_code == 200:
        user_data = response_user.json()
        animal_data = response_animal.json()

        # print(f"Usuarios cargados desde Firebase: {user_data.keys()}")
        # Ver estructura real
        # print(f"Datos completos de usuarios: {user_data}")
        # print(f"Animales cargados desde Firebase: {len(animal_data)}")

        if user_id in user_data:
            user_data_filtered = user_data[user_id]
        else:
            # print(f"⚠️ El usuario {user_id} no se encuentra en Firebase.")
            user_data_filtered = {}

        # print(f"Datos filtrados del usuario {user_id}: {user_data_filtered}")

        return animal_data, user_data_filtered
    else:
        print("❌ Error al cargar los datos desde Firebase")
        return {}, {}

# Función para codificar variables categóricas


def encode_categorical_variable(variable):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(variable)

# Preprocesamiento de datos de animales


def preprocess_animal_data(animal_data):
    features = []
    for animal in animal_data.values():
        edad = int(animal["edad"])
        raza = encode_categorical_variable([animal["raza"]])[0]
        sexo = encode_categorical_variable([animal["sexo"]])[0]
        recomendation = encode_categorical_variable(
            [animal["recomendation"]])[0]
        description = encode_categorical_variable([animal["descripcion"]])[0]

        features.append([edad, raza, sexo, recomendation, description])
    return np.array(features)

# Preprocesamiento de preferencias de usuarios
# user_id = 'YN41Eq0ObQU9OoepbDEDxnNPN4D2'


def preprocess_user_preferences(user_data, user_id):
    user_info = user_data if isinstance(user_data, dict) else {}

    # print(f"Datos completos del usuario {user_id}: {user_info}")

    if 'preferences' not in user_info or not user_info['preferences']:
        print(f"⚠️ El usuario {user_id} no tiene preferencias registradas.")
        return None

    prefs = user_info['preferences']
    # print(f"Preferencias extraídas: {prefs}")

    try:
        # Codificar las preferencias binarias
        encoded_prefs = [
            1 if prefs[key][0] == 'Sí' else 0 for key in [
                "aceptacionContrato", "compromisoEconomico", "disponibilidadSeguimiento",
                "espacioExterior", "experienciaMascota"
            ] if key in prefs
        ]

        # Codificar las preferencias categóricas
        categorical_prefs = [
            encode_categorical_variable([prefs[key][0]])[0] for key in [
                "foods", "hobbies", "moods", "motivacion", "preferenciaRaza",
                "selectedLifestyles", "situacionVivienda", "sports", "tiempoFuera",
                "tipoMascota", "ultimaMascota"
            ] if key in prefs
        ]

        return np.array([encoded_prefs + categorical_prefs])

    except Exception as e:
        print(f"❌ Error procesando preferencias del usuario: {e}")
        return None


# Preprocesamiento de imágenes de los animales


def preprocess_image(img_path):
    try:
        response = requests.get(img_path)
        if response.status_code == 200:
            img = image.load_img(BytesIO(response.content),
                                 target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
    except:
        return np.zeros((1, 224, 224, 3))

# Función para calcular la similitud


@register_keras_serializable()
def compute_similarity(vectors):
    user_vec, animal_vec = vectors
    return K.exp(-K.sum(K.abs(user_vec - animal_vec), axis=1, keepdims=True))

# Cargar el modelo entrenado


def load_model_from_path():
    model_path = 'Model/Best_Model_Siamese.keras'
    return load_model(model_path, custom_objects={'compute_similarity': compute_similarity})

# Función para predecir compatibilidad


# Función para predecir compatibilidad
def predict_compatibility(user_id, model, animal_data, user_data):
    # Preprocesar las preferencias del usuario
    user_features = preprocess_user_preferences(user_data, user_id)
    if user_features is None:
        return None

    # Preprocesar las características de los animales
    # Asegúrate de obtener las características de los animales aquí
    animal_features = preprocess_animal_data(animal_data)

    # Preprocesar las imágenes de los animales
    animal_images = np.array(
        [preprocess_image(animal["imagenes"][0]) for animal in animal_data.values()])
    animal_images = np.squeeze(animal_images, axis=1)

    # Expandir las características del usuario y los animales
    num_animals = len(animal_data)
    # Expandir las características del usuario
    user_features_expanded = np.tile(user_features, (num_animals, 1))
    animal_features_expanded = np.array(animal_features)

    # Redimensionar para que la forma coincida con la esperada por el modelo (771, 1)
    user_features_expanded = np.concatenate([user_features_expanded, np.zeros(
        (num_animals, 1165 - user_features_expanded.shape[1]))], axis=-1)
    animal_features_expanded = np.concatenate([animal_features_expanded, np.zeros(
        (num_animals, 771 - animal_features_expanded.shape[1]))], axis=-1)

    user_features_expanded = np.expand_dims(user_features_expanded, axis=-1)
    animal_features_expanded = np.expand_dims(
        animal_features_expanded, axis=-1)

    # Realizar la predicción para todos los animales
    similarity_scores = model.predict(
        [user_features_expanded, animal_features_expanded])

    # Retornar los resultados
    results = []
    for i, score in enumerate(similarity_scores):
        animal_id = list(animal_data.keys())[i]  # Obtener el ID del animal
        compatibility_percentage = score[0] * 100
        results.append((animal_id, compatibility_percentage))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results
