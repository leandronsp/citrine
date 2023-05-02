# frozen_string_literal: true

require 'matrix'
require_relative 'calc'
require_relative 'naive_matrix'

class BackPropagation
  def self.delta(target, output)
    error = Matrix[*target] - Matrix[*output]

    result = output.map do |array|
      array.map do |value|
        Calc.sigmoid_derivative(value)
      end
    end

    NaiveMatrix.new.multiply(result, error.to_a)
  end

  def self.delta_middle_layer(output, output_layer, delta_output_layer)
    factor = output_layer.to_matrix.transpose
    error = Matrix[*delta_output_layer] * Matrix[*factor]

    result = output.map do |array|
      array.map do |value|
        Calc.sigmoid_derivative(value)
      end
    end

    NaiveMatrix.new.multiply(result, error.to_a)
  end
end
