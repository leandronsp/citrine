
# frozen_string_literal: true

module Utils
  class Layer
    def self.build_layer(number_of_neurons:, number_of_weights:, random_weight: false)
      weight_func = -> { random_weight ? Random.rand(-1.0..1.0) : 1 }

      number_of_neurons
        .times
        .map { [weight_func.call] * number_of_weights }
        .map(&Models::Neuron.method(:new))
        .then(&Models::Layer.method(:new))
    end

    def self.build_layer_from_matrix(matrix)
      matrix
        .transpose
        .map(&Models::Neuron.method(:new))
        .then(&Models::Layer.method(:new))
    end

    def self.build_matrix_from_layer(layer)
      layer.neurons.map(&:weights).then(&:transpose)
    end
  end
end
