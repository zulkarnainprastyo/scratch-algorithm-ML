def logistic_regression(x, y, iterations=100,
                        learning_rate=0.01):
  m, n = len (x), len (x[0])
  beta_0, beta_other = initialize_params(n)
  for _ in range (iterations):
    gradient_beta_0, gradient_beta_other = (
        compute_gradients(x, y, beta_0,
                beta_other, m, n, 50))
    beta_0, beta_other = update_params(
        beta_0, beta_other, gradient_beta_0,
        gradient_beta_other, learning_rate)
    return beta_0, beta_other