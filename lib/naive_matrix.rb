class NaiveMatrix
  def multiply(matrix_one, matrix_two)
    case [matrix_one, matrix_two]
    in [[], []]
      return []
    else
      one_first_row = matrix_one.shift
      two_first_row = matrix_two.shift

      [do_multiply(one_first_row, two_first_row) +
        multiply(matrix_one, matrix_two)]
    end
  end

  private

  def do_multiply(array_one, array_two)
    case [array_one, array_two]
    in [[], []]
      return []
    else
      one_first = array_one.shift
      two_first = array_two.shift

      [one_first * two_first] + do_multiply(array_one, array_two)
    end
  end
end
