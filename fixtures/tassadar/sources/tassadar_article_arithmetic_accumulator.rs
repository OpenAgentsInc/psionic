#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn arithmetic_accumulator(seed: i32, rounds: i32) -> i32 {
    let mut total = seed.wrapping_mul(3).wrapping_add(7);
    let mut index = 0;
    while index < rounds {
        let lane = index & 3;
        if lane == 0 {
            total = total.wrapping_add(index.wrapping_mul(11).wrapping_add(3));
        } else if lane == 1 {
            total = total.wrapping_sub((seed ^ index).wrapping_mul(5));
        } else if lane == 2 {
            total ^= seed.wrapping_add(index << 1);
        } else {
            total = total.wrapping_mul(3).wrapping_add(index.wrapping_sub(seed));
        }
        index += 1;
    }
    total
}

#[unsafe(no_mangle)]
pub extern "C" fn arithmetic_mix_pair(left: i32, right: i32) -> i32 {
    let sum = left.wrapping_add(right);
    let diff = left.wrapping_sub(right);
    let weighted = left.wrapping_mul(3).wrapping_add(right.wrapping_mul(5));
    sum ^ diff ^ weighted
}
