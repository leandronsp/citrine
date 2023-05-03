class NeuralNetwork
  def initialize(layers)
    @layers = layers
  end

  def learn(inputs, targets, times)
    return if times <= 0

    outputs = ForwardPropagation.forward(@layers.dup, inputs)
    layers_and_outputs = @layers.zip(outputs)
    @layers = BackPropagation.adjust(inputs, targets, layers_and_outputs)

    learn(inputs, targets, times - 1)
  end

  def predict!(inputs)
    ForwardPropagation.predict!(@layers.dup, inputs)
  end
end
