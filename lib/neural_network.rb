class NeuralNetwork
  def initialize(layers)
    @layers = layers
  end

  def learn(inputs, targets, times)
    times.times do
      propagation_layers = ForwardPropagation::Forward.call(@layers.dup, inputs)
      @layers = BackPropagation::Adjust.call(inputs, targets, propagation_layers)
    end
  end

  def predict!(value)
    ForwardPropagation::Predict.call(@layers.dup, value)
  end
end
