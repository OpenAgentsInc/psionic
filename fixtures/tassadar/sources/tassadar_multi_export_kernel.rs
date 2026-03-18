#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn pair_sum() -> i32 {
    2 + 3
}

#[unsafe(no_mangle)]
pub extern "C" fn local_double() -> i32 {
    let seed = 7;
    seed + seed
}
