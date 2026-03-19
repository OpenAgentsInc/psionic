#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn add_one(value: i32) -> i32 {
    value + 1
}

#[unsafe(no_mangle)]
pub extern "C" fn pair_add(left: i32, right: i32) -> i32 {
    left + right
}
