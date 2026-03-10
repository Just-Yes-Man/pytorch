# Autoencoder 3-2-3 (PyTorch + Flask)

Este proyecto incluye:

- Entrenamiento de un autoencoder `3 -> 2 -> 3` en PyTorch.
- API Flask para recibir un array y devolver predicciones.
- Configuracion lista para desplegar en Render.

## 1) Entrenar y guardar el modelo

```bash
python ae_3_2_3_sigmoid_pytorch.py
```

Esto genera el archivo:

- `models/ae_3_2_3_sigmoid.pt`

## 2) Levantar API Flask local

```bash
pip install -r requirements.txt
python app.py
```

Endpoints:

- `GET /health`
- `POST /predict`

Ejemplo de request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
	-H "Content-Type: application/json" \
	-d '{"array": [[1,0,1],[0,1,0]]}'
```

Tambien acepta un solo vector:

```json
{"array": [1,0,1]}
```

## 3) Deploy en Render (Web Service)

1. Subir este proyecto a GitHub.
2. En Render: `New +` -> `Web Service`.
3. Conectar el repositorio.
4. Configurar:
	 - Runtime: `Python 3`
	 - Build Command: `pip install -r requirements.txt && python ae_3_2_3_sigmoid_pytorch.py`
	 - Start Command: `gunicorn app:app`
5. Deploy.

Opcional:

- Variable de entorno `MODEL_PATH` si quieres cargar el modelo desde otra ruta.
