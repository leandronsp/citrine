class NeuralNetwork
  def initialize(layers)
    @layers = layers
  end

  def learn(inputs, targets, times)
    times.times do
      layers_and_outputs = ForwardPropagation::Predict.call(@layers.dup, inputs)
      @layers = BackPropagation.adjust(inputs, targets, layers_and_outputs)
    end
  end

  def predict!(value)
    ForwardPropagation.predict!(@layers.dup, value)
  end
end
