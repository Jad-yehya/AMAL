import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
if torch.autograd.gradcheck(mse, (yhat, y)):
    print('OK')
else:
    print('Problème gradient MSE')

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)
X = torch.randn(50, 13, requires_grad=True, dtype=torch.float64)
W = torch.randn(13, 3, requires_grad=True, dtype=torch.float64)
b = torch.randn(3, requires_grad=True, dtype=torch.float64)
if torch.autograd.gradcheck(linear, (X, W, b)):
    print('OK')
else:
    print("Problème gradient Linear")
