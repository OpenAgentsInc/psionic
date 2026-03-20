#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const HEAP_WORDS: usize = 64;
static mut HEAP_BUFFER: [i32; HEAP_WORDS] = [0; HEAP_WORDS];

fn reset_heap() {
    let mut index = 0;
    while index < HEAP_WORDS {
        unsafe {
            HEAP_BUFFER[index] = 0;
        }
        index += 1;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bump_allocator_checksum(values: *const i32, len: i32) -> i32 {
    reset_heap();

    let mut cursor = 0usize;
    let mut index = 0;
    while index < len && cursor + 2 <= HEAP_WORDS {
        let value = unsafe { *values.add(index as usize) };
        unsafe {
            HEAP_BUFFER[cursor] = value;
            HEAP_BUFFER[cursor + 1] = value.wrapping_mul(index.wrapping_add(1));
        }
        cursor += 2;
        index += 1;
    }

    let mut checksum = 0i32;
    let mut read_index = 0usize;
    while read_index < cursor {
        checksum = checksum.wrapping_add(unsafe { HEAP_BUFFER[read_index] });
        read_index += 1;
    }

    checksum ^ cursor as i32
}
