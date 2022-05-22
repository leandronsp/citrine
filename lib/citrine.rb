class Citrine
  def sigmoid(number) = 1 / (1 + Math.exp(-number))

  def sigmoid_derivative(number) = number * (1 - number)
end
