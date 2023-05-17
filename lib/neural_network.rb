class NeuralNetwork
  def initialize(layers)
    @layers = layers
  end

  def learn(inputs, targets, times)
    times.times do
      layers_with_results = ForwardPropagation.call(@layers, inputs)
      @layers = BackPropagation.call(inputs, targets, layers_with_results)
    end
  end

  def predict(inputs) = Predict.call(@layers, inputs)
end
