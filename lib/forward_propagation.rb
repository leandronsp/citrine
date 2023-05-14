# frozen_string_literal: true

class ForwardPropagation
  def self.forward(*args) = new(*args).forward!
  def self.predict!(*args) = forward(*args).last.first

  def initialize(layers, inputs)
    @layers = layers
    @inputs = inputs
  end

  def forward!(inputs: @inputs, acc: [])
    return acc if @layers.empty?

    layer = @layers.shift

    result =
      (Matrix[*inputs] * Matrix[*layer.to_matrix])
      .map(&Calc.method(:sigmoid))

    forward!(inputs: result, acc: acc + [result])
  end
end
