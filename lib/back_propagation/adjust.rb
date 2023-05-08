# frozen_string_literal: true

module BackPropagation
  class Adjust
    def self.call(*args) = new(*args).call

    def initialize(inputs, target, layers_and_outputs)
      @inputs = inputs
      @target = target
      @layers_and_outputs = layers_and_outputs
    end

    def call
      @layers_and_outputs
        .map_with_index { |(layer, output), idx| adjust(layer, output, idx) }
    end

    def adjust(layer, output, idx)
      return adjust_input_layer(layer, output) if idx == 0
      return adjust_output_layer(layer, output) if idx == @layers_and_outputs.size - 1

      adjust_hidden_layer(layer, output, idx)
    end

    def adjust_input_layer(layer, output)
      adjust_layer(
        @inputs,
        delta_using_last_layer(output),
        layer
      )
    end

    def adjust_hidden_layer(layer, output, idx)
      output_of_previous_layer = @layers_and_outputs[idx - 1][1]

      adjust_layer(
        output_of_previous_layer,
        delta_using_last_layer(output),
        layer
      )
    end

    def adjust_output_layer(layer, output)
      output_of_previous_layer = @layers_and_outputs[-2][1]

      adjust_layer(
        output_of_previous_layer,
        delta(@target, output)
        layer
      )
    end

    def last_layer
      @last_layer ||= @layers_and_outputs[-1][0]
    end

    def output_of_last_layer
      @output_of_last_layer ||= @layers_and_outputs[-1[1]
    end

    def delta_of_last_layer
      @delta_of_last_layer ||= delta(@target, output_of_last_layer)
    end

    def adjust_layer(data, delta, layer)
      adjustment = Matrix[*data].transpose * Matrix[*delta]
      matrix = Matrix[*layer.to_matrix] + adjustment

      Layer.from_matrix(matrix.to_a)
    end

    def delta(target, output)
      error = Matrix[*target] - Matrix[*output]

      NaiveMatrix.new(
        output_with_activation(output),
        error.to_a
      ).multiply
    end

    def delta_using_last_layer(output)
      factor = last_layer.to_matrix.transpose
      error = Matrix[*delta_of_last_layer] * Matrix[*factor]

      NaiveMatrix.new(
        output_with_activation(output),
        error.to_a
      ).multiply
    end

    def output_with_activation(output)
      output.to_a.map do |array|
        array.map do |value|
          Calc.sigmoid_derivative(value)
        end
      end
    end
  end
end
