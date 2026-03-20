#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const STATE_IDLE: i32 = 0;
const STATE_READ: i32 = 1;
const STATE_BRANCH: i32 = 2;
const STATE_WRITE: i32 = 3;
const STATE_HALT: i32 = 4;

#[unsafe(no_mangle)]
pub extern "C" fn state_machine_router(steps: i32) -> i32 {
    let mut remaining = steps;
    let mut state = STATE_IDLE;
    let mut checksum = 0i32;
    while remaining > 0 {
        match state {
            STATE_IDLE => {
                checksum = checksum.wrapping_add(1);
                state = STATE_READ;
            }
            STATE_READ => {
                checksum = checksum.wrapping_mul(3).wrapping_add(remaining);
                if (remaining & 1) == 0 {
                    state = STATE_WRITE;
                } else {
                    state = STATE_BRANCH;
                }
            }
            STATE_BRANCH => {
                checksum ^= remaining.wrapping_mul(11);
                if remaining > 3 {
                    state = STATE_READ;
                } else {
                    state = STATE_HALT;
                }
            }
            STATE_WRITE => {
                checksum = checksum.rotate_left(1).wrapping_add(state);
                if remaining > 5 {
                    state = STATE_BRANCH;
                } else {
                    state = STATE_HALT;
                }
            }
            _ => {
                checksum = checksum.wrapping_add(remaining.wrapping_mul(7));
                state = STATE_IDLE;
            }
        }
        remaining -= 1;
    }
    checksum ^ state
}
