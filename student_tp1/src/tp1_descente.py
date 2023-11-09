import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    ctx = Context()
    ctx2 = Context()

    yhat = Linear.forward(ctx, x, w, b)

    loss = MSE.forward(ctx2, yhat, y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    grad_yhat, grad_y = MSE.backward(ctx2, 1)

    grad_x, grad_w, grad_b = Linear.backward(ctx, grad_yhat)


    ##  TODO:  Mise à jour des paramètres du modèle
    w = w - epsilon * grad_w
    b = b - epsilon * grad_b



