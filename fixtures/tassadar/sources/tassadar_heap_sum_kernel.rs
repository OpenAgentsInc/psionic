#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn heap_sum_i32(values: *const i32, len: i32) -> i32 {
    let mut total = 0;
    let mut index = 0;
    while index < len {
        total += unsafe { *values.add(index as usize) };
        index += 1;
    }
    total
}
