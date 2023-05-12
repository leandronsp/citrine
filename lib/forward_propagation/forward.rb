# frozen_string_literal: true

module ForwardPropagation
  class Forward
    def self.call(*args) = new(*args).call

    def initialize(layers, inputs)
      @propagation_layers = build_propagation_layers(layers)
      @inputs = inputs
    end

    def call
      forward!(@propagation_layers.dup, @inputs.dup)
    end

    private

    def build_propagation_layers(layers)
      layers.map { |layer| Models::PropagationLayer.new(layer) }
    end

    def forward!(propagation_layers, inputs, acc = [])
      return acc if propagation_layers.empty?

      propagation_layer = propagation_layers.shift

      propagation_layer.output =
        (Matrix[*inputs] * Matrix[*propagation_layer.to_matrix])
          .map(&Utils::ActivationFunction.method(:sigmoid))

      forward!(
        propagation_layers,
        propagation_layer.output,
        acc + [propagation_layer]
      )
    end
  end
end
