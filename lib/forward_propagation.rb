# frozen_string_literal: true

class ForwardPropagation
  def self.predict!(layers, inputs)
    forward(layers, inputs).last.first
  end

  def self.forward(layers, inputs)
    return inputs if layers.empty?

    result = predict_result(inputs, layers.shift)

    predict(result, layers, [result])
  end

  def self.predict(result, inputs, acc = [])
    return acc if inputs.empty?

    result = predict_result(result, inputs.shift)

    predict(result, inputs, acc + [result])
  end

  def self.predict_result(inputs, layer)
    (Matrix[*inputs] * Matrix[*layer.to_matrix])
      .map(&Calc.method(:sigmoid))
  end
end
