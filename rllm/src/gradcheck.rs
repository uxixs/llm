//! Test-only finite-difference gradient checker.

/// Compares `analytic` against central differences of `f` at `x` (eps 1e-2).
/// Per element: `|a - n| <= 1e-3 + 2e-2 * max(|a|, |n|)`.
pub fn check(f: impl Fn(&[f32]) -> f32, x: &[f32], analytic: &[f32]) {
    assert_eq!(x.len(), analytic.len(), "gradcheck: length mismatch");
    let eps = 1e-2f32;
    let mut xp = x.to_vec();
    for i in 0..x.len() {
        xp[i] = x[i] + eps;
        let hi = f(&xp);
        xp[i] = x[i] - eps;
        let lo = f(&xp);
        xp[i] = x[i];
        let numeric = (hi - lo) / (2.0 * eps);
        let a = analytic[i];
        let tol = 1e-3 + 2e-2 * a.abs().max(numeric.abs());
        assert!(
            (a - numeric).abs() <= tol,
            "gradcheck failed at index {i}: analytic {a}, numeric {numeric}"
        );
    }
}
