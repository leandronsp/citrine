class NeuralNetwork
  def initialize(layers)
    @layers = layers
  end

  def learn(inputs, targets, times)
    times.times do
      results = ForwardPropagation.forward(@layers.dup, inputs)
      layers_with_results = @layers.zip(results).map(&method(:enhance_layer))
      @layers = BackPropagation.adjust(inputs, targets, layers_with_results)
    end
  end

  def predict!(inputs)
    ForwardPropagation.predict!(@layers.dup, inputs)
  end

  private

  def enhance_layer((layer, result))
    layer.tap { |layer| layer.result = result }
  end
end
