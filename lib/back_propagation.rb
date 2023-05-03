# frozen_string_literal: true

require 'matrix'
require_relative 'calc'
require_relative 'naive_matrix'

class BackPropagation
  # First or Unique Layer
  def self.adjust(inputs, target, layers_and_outputs)
    layer, output = layers_and_outputs[0]
    remaining = layers_and_outputs[1..-1]

    return acc unless output

    if !remaining || remaining.empty?
      delta = delta(target, output)
      adjusted = apply_adjustment(inputs, delta, layer)

      return [adjusted]
    end

    last_layer, output_last_layer = remaining.last

    delta_last_layer = delta(target, output_last_layer)
    delta = delta_output_layer(output, last_layer, delta_last_layer)
    adjusted = apply_adjustment(inputs, delta, layer)

    adjust_middle_layer(inputs, target, remaining, output, [adjusted])
  end

  # Middle Layer
  def self.adjust_middle_layer(inputs, target, layers_and_outputs, output_previous_layer, acc)
    layer, output = layers_and_outputs[0]
    remaining = layers_and_outputs[1..-1]

    return acc unless output

    if !remaining || remaining.empty?
      delta = delta(target, output)
      adjusted = apply_adjustment(output_previous_layer, delta, layer)

      return acc + [adjusted]
    end

    last_layer, output_last_layer = remaning.last

    delta_last_layer = delta(target, output_last_layer)
    delta = delta_middle_layer(output, last_layer, delta_last_layer)
    adjusted = apply_adjustment(output_previous_layer, delta, layer)

    adjust_middle_layer(inputs, target, remaining, output, acc + [adjusted])
  end

  def self.apply_adjustment(inputs, delta, layer)
    adjustment = Matrix[*inputs].transpose * Matrix[*delta]
    matrix = Matrix[*layer.to_matrix] + adjustment

    Layer.from_matrix(matrix.to_a)
  end

  def self.delta(target, output)
    error = Matrix[*target] - Matrix[*output]

    result = output.to_a.map do |array|
      array.map do |value|
        Calc.sigmoid_derivative(value)
      end
    end

    NaiveMatrix.new(result.dup, error.to_a).multiply
  end

  def self.delta_output_layer(output, output_layer, delta_output_layer)
    factor = output_layer.to_matrix.transpose
    error = Matrix[*delta_output_layer] * Matrix[*factor]

    result = output.to_a.map do |array|
      array.map do |value|
        Calc.sigmoid_derivative(value)
      end
    end

    NaiveMatrix.new(result.dup, error.to_a).multiply
  end

  def self.delta_middle_layer(output, output_layer, delta_output_layer)
    factor = output_layer.to_matrix.transpose
    error = Matrix[*delta_output_layer] * Matrix[*factor]

    result = output.to_a.map do |array|
      array.map do |value|
        Calc.sigmoid_derivative(value)
      end
    end

    NaiveMatrix.new(result.dup, error.to_a).multiply
  end
end
