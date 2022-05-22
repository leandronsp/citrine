class Calc
  def self.sigmoid(number) = 1 / (1 + Math.exp(-number))

  def self.sigmoid_derivative(number) = number * (1 - number)
end
