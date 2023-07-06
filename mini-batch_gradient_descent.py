def compute_gradient_minibatch(
    x, y, beta_0, beta_other, m, n, batch_size):
    gradient_beta_0 = 0
    gradient_beta_other = [0] * n

    for _ in range(batch_size):
      i = random.randint(0, m - 1)
      point = x[i]
      pred = logistic_function(point, beta_o,
                               beta_other)
      for j, feature in enumerate(point):
        gradient_beta_other[j] +=(
            pred - y[i]) * feature / batch_size
      gradient_beta_0 += (pred - y[i]) / batch_size

    return gradient_beta_0, gradient_beta_other