from flask import Flask, jsonify, request, render_template_string
from model_utils import load_data_from_firebase, load_model_from_path, predict_compatibility

app = Flask(__name__)

# user_id = 'YN41Eq0ObQU9OoepbDEDxnNPN4D2'


@app.route('/')
def index():
    # HTML que incluye la imagen y un texto.
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API de Recomendación de Mascotas</title>
        <style>
            body {
                text-align: center;
                font-family: Arial, sans-serif;
            }
            img {
                width: 300px;
                height: auto;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Bienvenido a la API de Recomendación de Mascotas</h1>
        <img src="{{ url_for('static', filename='images/matchpet_logo.jpeg') }}" alt="Logo">
    </body>
    </html>
    """
    return render_template_string(html_content)


@app.route('/predict/<user_id>', methods=['GET'])
def predict(user_id):
    try:
        print(f"Request recibido para el user_id: {user_id}")

        # Cargar datos desde Firebase
        print("Cargando datos desde Firebase...")
        animal_data, user_data = load_data_from_firebase(user_id)
        print(load_data_from_firebase)

        if not animal_data or not user_data:
            return jsonify({"error": "Error al cargar los datos desde Firebase"}), 400

        # Cargar el modelo
        print("Cargando el modelo...")
        model = load_model_from_path()

        # Verificar que el modelo se ha cargado correctamente
        if model is None:
            return jsonify({"error": "Error al cargar el modelo"}), 500
        print("Modelo cargado exitosamente.")

        # Predecir compatibilidad
        print("Generando predicción de compatibilidad...")
        compatibility_results = predict_compatibility(
            user_id, model, animal_data, user_data)
        print(f"Resultados de compatibilidad: {compatibility_results}")

        if compatibility_results:
            return jsonify({"results": compatibility_results})
        else:
            return jsonify({"error": "No results found"}), 400

    except Exception as e:
        print(f"Error inesperado: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
