#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

static LOOKUP_TABLE: [i32; 3] = [11, 19, 23];

#[unsafe(no_mangle)]
pub extern "C" fn load_middle() -> i32 {
    LOOKUP_TABLE[1]
}

#[unsafe(no_mangle)]
pub extern "C" fn load_edge_sum() -> i32 {
    LOOKUP_TABLE[0] + LOOKUP_TABLE[2]
}
