#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn branch_dispatch_checksum(seed: i32) -> i32 {
    let mut value = seed;
    let mut index = 0i32;
    while index < 32 {
        let selector = (value ^ index.wrapping_mul(17)) & 7;
        if selector == 0 {
            value = value.wrapping_add(index.wrapping_mul(9).wrapping_add(3));
        } else if selector == 1 {
            value = value.wrapping_sub((index << 2).wrapping_add(5));
        } else if selector == 2 {
            value ^= 0x5A5A ^ index;
        } else if selector == 3 {
            value = value.rotate_left(1);
        } else if selector == 4 {
            value = value.rotate_right(1);
        } else if selector == 5 {
            value = value.wrapping_mul(3).wrapping_add(1);
        } else if selector == 6 {
            value ^= seed.wrapping_mul(index.wrapping_add(1));
        } else {
            value = value.wrapping_add((value & 3).wrapping_mul(7).wrapping_sub(index));
        }
        index += 1;
    }
    value
}
