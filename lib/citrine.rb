class Citrine
  attr_reader :layers

  def initialize(network_data)
    @layers = network_data.map(&method(:build_layer))
  end

  private

  def build_layer(layer_data)
    layer_data => [number_of_neurons, number_of_weights]

    Layer.build(
      number_of_neurons: number_of_neurons,
      number_of_weights: number_of_weights
    )
  end
end
