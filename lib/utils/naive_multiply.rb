# frozen_string_literal: true

module Utils
  class NaiveMultiply
    def self.call(*args) = new(*args).call

    def initialize(matrix_one, matrix_two)
      @matrix_one = matrix_one
      @matrix_two = matrix_two
    end

    def call
      @matrix_one.map.with_index do |array_one, idx|
        array_two = @matrix_two[idx]

        array_one.map.with_index do |value_one, idx|
          value_two = array_two[idx]

          value_one * value_two
        end
      end
    end
  end
end
