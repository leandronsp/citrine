# frozen_string_literal: true

require 'matrix'
require_relative 'calc'
require_relative 'naive_matrix_multiply'

class BackPropagation
  def self.adjust(*args) = new(*args).adjust

  def initialize(inputs, target, layers)
    @inputs = inputs
    @target = target
    @layers = layers
  end

  def adjust = @layers.map.with_index(&method(:adjust_layer))
  def output_layer = @layers.last
  def previous_layer_of(layer) = @layers[@layers.index(layer) - 1]

  def adjust_layer(layer, index)
    adjusted_layer(
      layer: layer,
      result: index.zero? ? @inputs : previous_layer_of(layer).result,
      delta: delta(layer)
    )
  end

  def delta(layer)
    return delta_output_layer if layer == output_layer

    factor = output_layer.to_matrix.transpose
    error = Matrix[*delta_output_layer] * Matrix[*factor]

    NaiveMatrixMultiply.call(result_activation(layer.result).dup, error.to_a)
  end

  def delta_output_layer
    @delta_output_layer ||=
      begin
        error = Matrix[*@target] - Matrix[*output_layer.result]
        NaiveMatrixMultiply.call(result_activation(output_layer.result).dup, error.to_a)
      end
  end

  def result_activation(result)
    result.to_a.map do |array|
      array.map do |value|
        Calc.sigmoid_derivative(value)
      end
    end
  end

  def adjusted_layer(layer:, result:, delta:)
    adjustment = Matrix[*result].transpose * Matrix[*delta]
    adjusted = Matrix[*layer.to_matrix] + adjustment

    Layer.from_matrix(adjusted.to_a)
  end
end
