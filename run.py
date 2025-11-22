#
import torch
import torch.nn as nn  # responsapor estrutura da rede neural
import torch.optim as optim  # responsapor otimizacao da rede neural

# dados que o modelo vai usar para treinar
X = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)
# resultados esperados para os dados de entrada
y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)

# definicao da rede neural


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # primeira camada de entrada com 1 neurônio e 5 neurônios na camada oculta
        self.fcl = nn.Linear(1, 5)
        # segunda camada de saida com 5 neurônios na camada oculta e 1 neurônio na camada de saida
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fcl(x))
        x = self.fc2(x)
        return x


model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# treinamento da rede neural. Faz previsoes e ajusta os pesos da rede neural para minimizar o erro.
for epoch in range(1000):
    optimizer.zero_grad()  # zera os gradientes
    output = model(X)  # faz a previsao
    loss = criterion(output, y)  # calcula o erro
    loss.backward()  # calcula o gradiente do erro
    optimizer.step()  # ajusta os pesos da rede neural

    if epoch % 100 == 99:  # imprime o erro a cada 100 epocas
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

with torch.no_grad():  # nao calcula os gradientes
    # faz a previsao para um novo dado
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(f'Previsao do tempo de conclusao: {predicted.item()} minutos')
