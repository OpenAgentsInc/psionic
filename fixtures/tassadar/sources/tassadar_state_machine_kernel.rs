#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const ITERATION_COUNT: i32 = 2_000_000;

#[unsafe(no_mangle)]
pub extern "C" fn state_machine_loop() -> i32 {
    let mut remaining = ITERATION_COUNT;
    let mut state = 0i32;
    let mut accumulator = 0i32;
    while remaining > 0 {
        state = 1 - state;
        accumulator += state;
        remaining -= 1;
    }
    accumulator
}
