def forward_pass(x, w, b):
    y_pred = x * w + b
    return y_pred

def compute_loss(y_pred, y_true):
    loss = (abs(y_pred - y_true)) ** 2
    return loss

def compute_gradients(x , y_pred, y_true):
    gred_w = 2 * (y_pred - y_true) * x
    gred_b = 2 * (y_pred - y_true)
    return gred_w,gred_b

def update_parameters(w, b, grad_w, grad_b, lr):
    w_new = w - lr * gred_w
    b_new = b - lr * gred_b
    return w_new,b_new
x = 2.0
w = 1.0
b = 0.0 
y_true = 5.0
lr = 0.05
for i in range(20):
    y_pred = forward_pass(x, w, b)
    loss = compute_loss(y_pred, y_true)
    (gred_w, gred_b) = compute_gradients(x, y_pred, y_true)
    (w, b) = update_parameters(w, b, gred_w, gred_b, lr)
    print(loss)