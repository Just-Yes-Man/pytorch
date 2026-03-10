import os

import torch
from flask import Flask, jsonify, request

from pytorch_model import Autoencoder


MODEL_PATH = os.environ.get("MODEL_PATH", "models/ae_3_2_3_sigmoid.pt")

app = Flask(__name__)


def load_model(path: str) -> Autoencoder:
    model = Autoencoder()
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontro el modelo en '{MODEL_PATH}'. Ejecuta primero ae_3_2_3_sigmoid_pytorch.py"
    )

model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload or "array" not in payload:
        return jsonify({"error": "Body JSON invalido. Usa {'array': [[1,0,1]]}"}), 400

    arr = payload["array"]
    if not isinstance(arr, list) or len(arr) == 0:
        return jsonify({"error": "'array' debe ser una lista no vacia"}), 400

    # Acepta [1,0,1] y tambien [[1,0,1], ...]
    if isinstance(arr[0], (int, float)):
        arr = [arr]

    try:
        x = torch.tensor(arr, dtype=torch.float32)
    except Exception:
        return jsonify({"error": "No se pudo convertir 'array' a tensor numerico"}), 400

    if x.ndim != 2 or x.shape[1] != 3:
        return jsonify({"error": "Forma invalida. Se espera (n, 3)"}), 400

    with torch.no_grad():
        y = model(x)

    return jsonify({"predictions": y.numpy().tolist()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
