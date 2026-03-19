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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dot_i32(left: *const i32, right: *const i32, len: i32) -> i32 {
    let mut total = 0;
    let mut index = 0;
    while index < len {
        total += unsafe { *left.add(index as usize) * *right.add(index as usize) };
        index += 1;
    }
    total
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sum_and_max_into_buffer(
    values: *const i32,
    len: i32,
    out: *mut i32,
    out_len: i32,
) -> i32 {
    if out_len < 2 {
        return 1;
    }

    let mut total = 0;
    let mut maximum = 0;
    let mut index = 0;
    while index < len {
        let value = unsafe { *values.add(index as usize) };
        total += value;
        if index == 0 || value > maximum {
            maximum = value;
        }
        index += 1;
    }

    unsafe {
        *out.add(0) = total;
        *out.add(1) = maximum;
    }
    0
}
