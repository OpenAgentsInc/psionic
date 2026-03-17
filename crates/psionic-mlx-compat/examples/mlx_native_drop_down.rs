use psionic_core::Shape;
use psionic_mlx_compat::core;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = core::cpu_seeded(11)?;
    let facade_values = context.array(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0])?;
    let native_bias = context.native().full_f32(Shape::new(vec![2, 2]), 1.0)?;
    let reduced = facade_values.add(&native_bias)?.sum_axis(1)?;
    let host = reduced.eval()?.to_host_data()?;

    assert_eq!(host.as_f32_slice(), Some(&[5.0, 9.0][..]));
    assert_eq!(context.device_handle().backend(), "cpu");

    Ok(())
}
