use psionic_core::Shape;
use psionic_mlx_compat::{core, reports};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = core::cpu_seeded(7)?;
    let lhs = context.ones(Shape::new(vec![2, 2]))?;
    let rhs = context.array(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0])?;
    let reduced = lhs.add(&rhs)?.sum_axis(1)?;
    let host = reduced.eval()?.to_host_data()?;

    assert_eq!(host.as_f32_slice(), Some(&[5.0, 9.0][..]));

    let report = reports::builtin_mlx_compatibility_matrix_report();
    let array_surface = report
        .surfaces
        .iter()
        .find(|surface| surface.surface_id == "public_mlx_array_api")
        .ok_or("missing `public_mlx_array_api` compatibility row")?;
    assert_eq!(
        array_surface.matrix_status,
        reports::MlxCompatibilityMatrixStatus::Supported
    );

    Ok(())
}
