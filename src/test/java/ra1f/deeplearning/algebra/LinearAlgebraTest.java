package ra1f.deeplearning.algebra;

import io.vavr.collection.List;
import org.junit.Test;

import static io.vavr.API.*;
import static junit.framework.Assert.assertEquals;
import static org.assertj.core.api.Assertions.assertThat;
import static ra1f.deeplearning.algebra.LinearAlgebra.*;

public class LinearAlgebraTest {

  @Test(expected = NullPointerException.class)
  public void dotX_null_NPE() {
    dotX(null, null);
    dotX(List(), null);
    dotX(null, List());
  }

  @Test
  public void dotX_empty_None() {
    assertEquals(None(), dotX(List(), List()));
    assertEquals(None(), dotX(List(1d), List()));
    assertEquals(None(), dotX(List(), List(1d)));
  }

  @Test
  public void dotX_noneEmpty_Some() {
    assertEquals(Some(4d), dotX(List(1d), List(4d)));
    assertEquals(Some(14d), dotX(List(1d, 2d), List(4d, 5d)));
    assertEquals(Some(32d), dotX(List(1d, 2d, 3d), List(4d, 5d, 6d)));
    assertEquals(Some(32d), dotX(List(1d, 2d, 3d), List(4d, 5d, 6d, 7d)));
    assertEquals(Some(32d), dotX(List(1d, 2d, 3d, 4d), List(4d, 5d, 6d)));
  }

  @Test(expected = NullPointerException.class)
  public void mXv_null_NPE() {
    mXv(null, null);
    mXv(List(), null);
    mXv(null, List());
  }

  @Test
  public void mXv_empty_Empty() {
    assertEquals(List(), mXv(List(), List()));
    assertEquals(List(), mXv(List(List(1d)), List()));
    assertEquals(List(), mXv(List(), List(1d)));
  }

  @Test
  public void mXv_nonEmpty_Some() {
    assertEquals(List(1d), mXv(List(List(1d)), List(1d)));
    assertEquals(List(2d, 2d),
        mXv(List(List(1d, 1d), List(1d, 1d)), List(1d, 1d)));
    assertEquals(List(68d, 84d, 87d),
        mXv(List(List(2d, 4d, 6d), List(6d, 2d, 8d), List(7d, 9d, 3d)),
            List(3d, 5d, 7d)));
  }

  @Test(expected = NullPointerException.class)
  public void matrix_null_NPE() {
    matrix(4, 3, null);
  }

  @Test
  public void matrix_valid_Matrix() {
    List randomMatrix = matrix(4, 3, () -> Math.random() - .5);
    randomMatrix
        .flatMap(n -> n)
        .forEach(val -> assertThat((Double)val).isBetween(-0.5d, 0.5d));
    assertThat(randomMatrix.size()).isEqualTo(4);
    assertThat(randomMatrix.flatMap(n -> n).size()).isEqualTo(4 * 3);
  }

  @Test
  public void matrix_empty_Empty() {
    assertThat(matrix(0, 0, () -> 0d)).isEmpty();
    assertThat(matrix(0, 1, () -> 0d)).isEmpty();
    matrix(1, 0, () -> 0d).forEach(row -> assertThat(row).isEmpty());

  }
}
