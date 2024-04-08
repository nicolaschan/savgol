use crate::math;

pub struct Filter {
    radius: usize,   // m
    degree: u64,     // n
    derivative: u64, // s
}

impl Filter {
    pub fn new(radius: usize, degree: u64, derivative: u64) -> Self {
        Filter {
            radius,
            degree,
            derivative,
        }
    }

    fn weight_uncached(&self, i: i64, t: i64) -> f64 {
        math::weights(
            i,
            self.radius as i64,
            self.degree as i64,
            t,
            self.derivative as i64,
        )
    }

    fn weight(&self, i: i64, t: i64) -> f64 {
        self.weight_uncached(i, t)
    }

    /// Make sure you have a window of size 2 * RADIUS + 1
    fn smooth_point(&self, t: i64, window: &[f64]) -> f64 {
        assert!(window.len() == 2 * self.radius + 1);
        let radius = self.radius as isize;

        let mut sum = 0.0;
        for i in -radius..=radius {
            sum += self.weight(i as i64, t) * window[(i + radius) as usize];
        }
        sum
    }

    fn smooth_edge(&self, start_t: isize, end_t: isize, window: &[f64]) -> Vec<f64> {
        let mut smoothed = Vec::new();
        for t in start_t..=end_t {
            smoothed.push(self.smooth_point(t as i64, window));
        }
        smoothed
    }

    pub fn smooth(&self, data: &[f64]) -> Vec<f64> {
        if data.len() <= 2 {
            return data.to_vec();
        }
        if data.len() < 2 * self.radius + 1 {
            let radius = (data.len() - 1) / 2;
            return Filter::new(radius, self.degree, self.derivative).smooth(data);
        }
        let mut smoothed = Vec::new();
        smoothed.extend(self.smooth_edge(
            -(self.radius as isize),
            -1,
            data[0..2 * self.radius + 1].as_ref(),
        ));
        for i in self.radius..data.len() - self.radius {
            let window = &data[i - self.radius..=i + self.radius];
            smoothed.push(self.smooth_point(0, window));
        }
        smoothed.extend(self.smooth_edge(
            1,
            self.radius as isize,
            data[data.len() - 2 * self.radius - 1..].as_ref(),
        ));
        smoothed
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    fn assert_float_eq(a: f64, b: f64) {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }

    #[test]
    fn smooth_two_points_is_unchanged() {
        let filter = super::Filter::new(1, 2, 0);
        let smoothed = filter.smooth(&[1.0, 2.0]);
        assert_eq!(smoothed, vec![1.0, 2.0]);
    }

    #[test]
    fn smooth_5pt_quadratic_on_7pts_linear() {
        let filter = super::Filter::new(2, 2, 0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let smoothed = filter.smooth(data.as_slice());
        assert_eq!(smoothed.len(), 7);
        assert_float_eq(smoothed[0], 1.0);
        assert_float_eq(smoothed[1], 2.0);
        assert_float_eq(smoothed[2], 3.0);
        assert_float_eq(smoothed[3], 4.0);
        assert_float_eq(smoothed[4], 5.0);
        assert_float_eq(smoothed[5], 6.0);
        assert_float_eq(smoothed[6], 7.0);
    }

    #[test]
    fn smooth_5pt_quadratic_on_7pts_nonlinear() {
        let filter = super::Filter::new(2, 2, 0);
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0];
        let smoothed = filter.smooth(data.as_slice());
        assert_eq!(smoothed.len(), 7);
        assert_float_eq(smoothed[0], 1.1142857142857143);
        assert_float_eq(smoothed[1], -0.8571428571428571);
        assert_float_eq(smoothed[2], -1.1142857142857143);
        assert_float_eq(smoothed[3], 1.4857142857142858);
        assert_float_eq(smoothed[4], -1.8571428571428572);
        assert_float_eq(smoothed[5], 0.17142857142857143);
        assert_float_eq(smoothed[6], 5.057142857142857);
    }

    #[test]
    fn smooth_5pt_quadratic_on_5pts_nonlinear_radius_too_large() {
        let filter = super::Filter::new(20, 2, 0);
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let smoothed = filter.smooth(data.as_slice());
        assert_eq!(smoothed.len(), 5);
        assert_float_eq(smoothed[0], 1.1142857142857143);
        assert_float_eq(smoothed[1], -0.8571428571428571);
        assert_float_eq(smoothed[2], -1.1142857142857143);
        assert_float_eq(smoothed[3], 0.34285714285714286);
        assert_float_eq(smoothed[4], 3.5142857142857142);
    }

    #[test]
    fn smooth_5pt_quadratic_on_6pts_nonlinear_radius_too_large() {
        let filter = super::Filter::new(20, 2, 0);
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let smoothed = filter.smooth(data.as_slice());
        assert_eq!(smoothed.len(), 6);
        assert_float_eq(smoothed[0], 1.1142857142857143);
        assert_float_eq(smoothed[1], -0.8571428571428571);
        assert_float_eq(smoothed[2], -1.1142857142857143);
        assert_float_eq(smoothed[3], 1.4857142857142858);
        assert_float_eq(smoothed[4], -0.2571428571428571);
        assert_float_eq(smoothed[5], -4.285714285714286);
    }

    #[test]
    fn smooth_point_5pt_quadratic_t_neg2_linear() {
        let filter = super::Filter::new(2, 2, 0);
        let smoothed = filter.smooth_point(-2, vec![1.0, 2.0, 3.0, 4.0, 5.0].as_slice());
        assert_float_eq(smoothed, 1.0);
    }

    #[test]
    fn smooth_point_5pt_quadratic_t_neg1_linear() {
        let filter = super::Filter::new(2, 2, 0);
        let smoothed = filter.smooth_point(-1, vec![1.0, 2.0, 3.0, 4.0, 5.0].as_slice());
        assert_float_eq(smoothed, 2.0);
    }

    #[test]
    fn smooth_point_5pt_quadratic_t_0_linear() {
        let filter = super::Filter::new(2, 2, 0);
        let smoothed = filter.smooth_point(0, vec![1.0, 2.0, 3.0, 4.0, 5.0].as_slice());
        assert_float_eq(smoothed, 3.0);
    }

    #[test]
    fn smooth_point_5pt_quadratic_t_1_linear() {
        let filter = super::Filter::new(2, 2, 0);
        let smoothed = filter.smooth_point(1, vec![1.0, 2.0, 3.0, 4.0, 5.0].as_slice());
        assert_float_eq(smoothed, 4.0);
    }

    #[test]
    fn smooth_point_5pt_quadratic_t_2_linear() {
        let filter = super::Filter::new(2, 2, 0);
        let smoothed = filter.smooth_point(2, vec![1.0, 2.0, 3.0, 4.0, 5.0].as_slice());
        assert_float_eq(smoothed, 5.0);
    }

    #[test]
    fn smooth_point_5pt_quadratic_t_neg2_nonlinear() {
        let filter = super::Filter::new(2, 2, 0);
        let smoothed = filter.smooth_point(-2, vec![1.0, -2.0, 3.0, -4.0, 5.0].as_slice());
        assert_float_eq(smoothed, 1.1142857142857143);
    }
}
