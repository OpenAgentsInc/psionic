#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const ITERATION_COUNT: i32 = 2_000_000;

#[unsafe(no_mangle)]
pub extern "C" fn million_step_loop() -> i32 {
    let mut remaining = ITERATION_COUNT;
    while remaining > 0 {
        remaining -= 1;
    }
    0
}
