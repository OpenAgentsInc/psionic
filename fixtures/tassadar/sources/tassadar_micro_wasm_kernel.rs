#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn micro_wasm_kernel(values: *const i32) -> i32 {
    let a = unsafe { *values.add(0) };
    let b = unsafe { *values.add(1) };
    let c = unsafe { *values.add(2) };
    let d = unsafe { *values.add(3) };
    a + (b * 2) + (c * 3) + (d * 4)
}
