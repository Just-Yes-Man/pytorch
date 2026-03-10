import torch
import torch.nn as nn
import torch.optim as optim
import os

from pytorch_model import Autoencoder

# -----------------------------
# Datos
# -----------------------------
x = torch.tensor([[1.,0.,1.]])

model = Autoencoder()

# -----------------------------
# Entrenamiento
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

for epoch in range(200):

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(output, x)

    loss.backward()

    optimizer.step()

    print("Epoch:", epoch, "Error:", loss.item())

# -----------------------------
# Resultado
# -----------------------------
print("\nEntrada original:", x)
print("Reconstrucción:", model(x))

# -----------------------------
# Guardado de pesos
# -----------------------------
model_path = "models/ae_3_2_3_sigmoid.pt"
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_path)
print("Pesos guardados en:", model_path)
