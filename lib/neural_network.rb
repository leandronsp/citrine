class NeuralNetwork
  def initialize(layers)
    @layers = layers
  end

  def predict!(inputs)
    ForwardPropagation.predict!(@layers, inputs)
  end
end
