require 'byebug'

class ForwardPropagation
  def self.predict(layers, inputs, acc = [])
    return acc if inputs.empty?
    return forward(layers, inputs, acc) if acc.empty?

    result = predict_result(layers, inputs.shift)

    inputs.shift
    predict(result, inputs, acc + [result])
  end

  def self.forward(layers, inputs, acc)
    return inputs if layers.empty?

    next_layer = layers.shift
    result     = predict_result(inputs, next_layer)

    predict(result, layers, [result])
  end

  def self.predict_result(inputs, layer)
    (Matrix[*inputs] * Matrix[*layer.to_matrix])
      .map(&Calc.method(:sigmoid))
  end
end
