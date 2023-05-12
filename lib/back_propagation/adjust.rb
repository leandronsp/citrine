# frozen_string_literal: true

module BackPropagation
  class Adjust
    def self.call(*args) = new(*args).call

    def initialize(inputs, target, propagation_layers)
      @inputs = inputs
      @target = target
      @propagation_layers = propagation_layers
    end

    def call
      @propagation_layers
        .map.with_index { |propagation_layer, idx| adjust(propagation_layer, idx) }
    end

    private

    def adjust(propagation_layer, idx)
      return adjust_input_layer(propagation_layer) if idx == 0
      return adjust_output_layer(propagation_layer) if idx == @propagation_layers.size - 1

      adjust_hidden_layer(propagation_layer, idx)
    end

    def adjust_input_layer(propagation_layer)
      propagation_layer.adjust!(
        inputs: @inputs,
        target: @target,
        output_layer: output_layer
      )
    end

    def adjust_hidden_layer(propagation_layer, idx)
      propagation_layer.adjust!(
        inputs: previous_layer(idx).output,
        target: @target,
        output_layer: output_layer
      )
    end

    def adjust_output_layer(propagation_layer)
      propagation_layer.adjust!(
        inputs: previous_layer(-1).output,
        target: @target
      )
    end

    def previous_layer(idx)
      @propagation_layers[idx - 1]
    end

    def output_layer
      @output_layer ||= @propagation_layers[-1]
    end
  end
end
