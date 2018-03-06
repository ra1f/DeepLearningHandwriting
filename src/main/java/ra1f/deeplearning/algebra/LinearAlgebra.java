package ra1f.deeplearning.algebra;

import io.vavr.collection.Iterator;
import io.vavr.collection.List;
import io.vavr.control.Option;
import lombok.NonNull;

import java.util.function.Supplier;

import static io.vavr.API.*;

public class LinearAlgebra {

  /**
   * Creates a mXn matrix
   * @param rows - count of rows of the matrix
   * @param cols - count of columns of the matrix
   * @param field - field initialization function
   * @return the matrix
   */
  public static List<List<Double>> matrix(int rows, int cols, @NonNull Supplier<Double> field) {
    return List.fill(rows, () -> List.fill(cols, field));
  }

  /**
   * Calculates the dot product for two vectors v1 and v2.
   * e.g.
   *                   (v21)
   * (v11, v12, v13) X (v22) = v11 * v21 + v12 * v22 + v13 * v23
   *                   (v23)
   *
   * @param v1 - a vector
   * @param v2 - a vector
   * @return a scalar
   */
  public static Option<Double> dotX(@NonNull List<Double> v1, @NonNull List<Double> v2) {

    return Match(For(v1.zip(v2)).yield(t -> t._1 * t._2)).of(
        Case($(Iterator::isEmpty), None()),
        Case($(), ps -> Some(ps.reduceLeft((l, r) -> l + r)))
    );
  }

  /**
   * Calculates the product of a matrix and a vector.
   * e.g.
   * (m11, m12, m13)   (v1)   (m11 * v1 + m12 * v2 + m13 * v3)
   * (m21, m22, m23) X (v2) = (m21 * v1 + m22 * v2 + m23 * v3)
   * (m31, m32, m33)   (v3)   (m31 * v1 + m32 * v2 + m33 * v3)
   *
   * @param m - a matrix
   * @param v - a vector
   * @return a vector
   */
  public static List<Double> mXv(@NonNull List<List<Double>> m, @NonNull List<Double> v) {

    return For(m)
        .yield(mRow -> dotX(mRow, v))
        .toList()
        .flatMap(o -> o);
  }
}
