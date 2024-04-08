// Reference: A., Gorry (1990). "General least-squares smoothing and differentiation by the convolution (Savitzky–Golay) method". Analytical Chemistry. 62 (6): 570–3. doi:10.1021/ac00205a007.

/// Calculates the ln generalized factorial (a)(a-1)...(a-b+1)
fn ln_generalized_factorial(a: i64, b: i64) -> f64 {
    statrs::function::factorial::ln_factorial(a as u64)
        - statrs::function::factorial::ln_factorial((a - b) as u64)
}

/// Calculates the Gram Polynomial (s=0), or it's s'th derivative
/// evaluated at i, order k, over 2m+1 points.
fn gram_poly(i: i64, m: i64, k: i64, s: i64) -> f64 {
    if k == 0 && s == 0 {
        return 1.0;
    }
    if k <= 0 {
        return 0.0;
    }

    let part1 = (4 * k - 2) as f64 / (k * (2 * m - k + 1)) as f64
        * (gram_poly(i, m, k - 1, s) * i as f64 + gram_poly(i, m, k - 1, s - 1) * s as f64);
    let part2 =
        ((k - 1) * (2 * m + k)) as f64 / (k * (2 * m - k + 1)) as f64 * gram_poly(i, m, k - 2, s);
    return part1 - part2;
}

/// Calculates the weight of the i'th data point for the t'th Least-Square
/// point of the s'th derivative, over 2m+1 points, order n.
pub fn weights(i: i64, m: i64, n: i64, t: i64, s: i64) -> f64 {
    let mut sum = 0.0;
    for k in 0..=n {
        sum += (2 * k + 1) as f64
            * (ln_generalized_factorial(2 * m, k) - ln_generalized_factorial(2 * m + k + 1, k + 1))
                .exp()
            * gram_poly(i, m, k, 0)
            * gram_poly(t, m, k, s);
    }
    sum
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn assert_float_eq(a: f64, b: f64) {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }

    #[test]
    fn generalized_factorial_4_2() {
        // a = 4, b = 2
        // (4)...(4-2+1) = 4*3 = 12
        assert_float_eq(ln_generalized_factorial(4, 2).exp(), 12.0);
    }

    #[test]
    fn generalized_factorial_5_5() {
        // a = 5, b = 5
        // (5)...(5-5+1) = 5*4*3*2*1 = 120
        assert_float_eq(ln_generalized_factorial(5, 5).exp(), 120.0);
    }

    #[test]
    fn generalized_factorial_5_0() {
        // a = 5, b = 0
        // (5)...(5-0+1) = 1
        assert_float_eq(ln_generalized_factorial(5, 0).exp(), 1.0);
    }

    #[test]
    fn generalized_factorial_5_1() {
        // a = 5, b = 1
        // (5)...(5-1+1) = 5
        assert_float_eq(ln_generalized_factorial(5, 1).exp(), 5.0);
    }

    #[test]
    fn weight_5pt_quadratic_t_0() {
        assert_float_eq(weights(-2, 2, 2, 0, 0), -3.0 / 35.0);
        assert_float_eq(weights(-1, 2, 2, 0, 0), 12.0 / 35.0);
        assert_float_eq(weights(0, 2, 2, 0, 0), 17.0 / 35.0);
        assert_float_eq(weights(1, 2, 2, 0, 0), 12.0 / 35.0);
        assert_float_eq(weights(2, 2, 2, 0, 0), -3.0 / 35.0);
    }

    #[test]
    fn weight_5pt_quadratic_t_neg2() {
        assert_float_eq(weights(-2, 2, 2, -2, 0), 31.0 / 35.0);
        assert_float_eq(weights(-1, 2, 2, -2, 0), 9.0 / 35.0);
        assert_float_eq(weights(0, 2, 2, -2, 0), -3.0 / 35.0);
        assert_float_eq(weights(1, 2, 2, -2, 0), -5.0 / 35.0);
        assert_float_eq(weights(2, 2, 2, -2, 0), 3.0 / 35.0);
    }

    #[test]
    fn weight_7pt_quadratic_t_2() {
        assert_float_eq(weights(-3, 3, 2, 2, 0), -1.0 / 14.0);
        assert_float_eq(weights(-2, 3, 2, 2, 0), 0.0 / 14.0);
        assert_float_eq(weights(-1, 3, 2, 2, 0), 1.0 / 14.0);
        assert_float_eq(weights(0, 3, 2, 2, 0), 2.0 / 14.0);
        assert_float_eq(weights(1, 3, 2, 2, 0), 3.0 / 14.0);
        assert_float_eq(weights(2, 3, 2, 2, 0), 4.0 / 14.0);
        assert_float_eq(weights(3, 3, 2, 2, 0), 5.0 / 14.0);
    }

    #[test]
    fn weight_5pt_first_deriv_quadratic_t_0() {
        assert_float_eq(weights(-2, 2, 2, 0, 1), -2.0 / 10.0);
        assert_float_eq(weights(-1, 2, 2, 0, 1), -1.0 / 10.0);
        assert_float_eq(weights(0, 2, 2, 0, 1), 0.0 / 10.0);
        assert_float_eq(weights(1, 2, 2, 0, 1), 1.0 / 10.0);
        assert_float_eq(weights(2, 2, 2, 0, 1), 2.0 / 10.0);
    }

    #[test]
    fn weight_7pt_first_deriv_quadratic_t_2() {
        assert_float_eq(weights(-3, 3, 2, 2, 1), 11.0 / 84.0);
        assert_float_eq(weights(-2, 3, 2, 2, 1), -6.0 / 84.0);
        assert_float_eq(weights(-1, 3, 2, 2, 1), -15.0 / 84.0);
        assert_float_eq(weights(0, 3, 2, 2, 1), -16.0 / 84.0);
        assert_float_eq(weights(1, 3, 2, 2, 1), -9.0 / 84.0);
        assert_float_eq(weights(2, 3, 2, 2, 1), 6.0 / 84.0);
        assert_float_eq(weights(3, 3, 2, 2, 1), 29.0 / 84.0);
    }

    #[test]
    fn weight_7pt_first_deriv_quadratic_t_neg1() {
        assert_float_eq(weights(-3, 3, 2, -1, 1), -19.0 / 84.0);
        assert_float_eq(weights(-2, 3, 2, -1, 1), -6.0 / 84.0);
        assert_float_eq(weights(-1, 3, 2, -1, 1), 3.0 / 84.0);
        assert_float_eq(weights(0, 3, 2, -1, 1), 8.0 / 84.0);
        assert_float_eq(weights(1, 3, 2, -1, 1), 9.0 / 84.0);
        assert_float_eq(weights(2, 3, 2, -1, 1), 6.0 / 84.0);
        assert_float_eq(weights(3, 3, 2, -1, 1), -1.0 / 84.0);
    }

    #[test]
    fn weight_7pt_quadratic_t_3() {
        assert_float_eq(weights(-3, 3, 2, 3, 0), 5.0 / 42.0);
        assert_float_eq(weights(-2, 3, 2, 3, 0), -3.0 / 42.0);
        assert_float_eq(weights(-1, 3, 2, 3, 0), -6.0 / 42.0);
        assert_float_eq(weights(0, 3, 2, 3, 0), -4.0 / 42.0);
        assert_float_eq(weights(1, 3, 2, 3, 0), 3.0 / 42.0);
        assert_float_eq(weights(2, 3, 2, 3, 0), 15.0 / 42.0);
        assert_float_eq(weights(3, 3, 2, 3, 0), 32.0 / 42.0);
    }

    #[test]
    fn weight_5pt_cubic_t_neg2() {
        assert_float_eq(weights(-2, 2, 3, -2, 0), 69.0 / 70.0);
        assert_float_eq(weights(-1, 2, 3, -2, 0), 4.0 / 70.0);
        assert_float_eq(weights(0, 2, 3, -2, 0), -6.0 / 70.0);
        assert_float_eq(weights(1, 2, 3, -2, 0), 4.0 / 70.0);
        assert_float_eq(weights(2, 2, 3, -2, 0), -1.0 / 70.0);
    }
}
