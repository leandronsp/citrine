# frozen_string_literal: true

class ForwardPropagation
  def self.call(*args) = new(*args).call

  def initialize(layers, inputs)
    @layers = layers.dup
    @inputs = inputs
  end

  def call
    @layers.map.with_index do |layer, index|
      data = index.zero? ? @inputs : @layers[index - 1].result

      layer.tap do
        layer.result = (Matrix[*data] * Matrix[*layer.to_matrix]).map(&Calc.method(:sigmoid))
      end
    end
  end
end
