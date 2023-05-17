# frozen_string_literal: true

require_relative './forward_propagation'

class Predict
  def self.call(*args) = new(*args).call

  def initialize(layers, inputs)
    @layers = layers.dup
    @inputs = inputs
  end

  def call
    layers_with_results = ForwardPropagation.call(@layers, @inputs)
    output_layer_result = layers_with_results.last.result

    output_layer_result.first
  end
end
