#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const DIM: usize = 4;
const COST_MATRIX: [i32; DIM * DIM] = [
    9, 2, 7, 8, //
    6, 4, 3, 7, //
    5, 8, 1, 8, //
    7, 6, 9, 4,
];

fn row_min(row: usize) -> i32 {
    let mut best = COST_MATRIX[row * DIM];
    let mut column = 1;
    while column < DIM {
        let value = COST_MATRIX[row * DIM + column];
        if value < best {
            best = value;
        }
        column += 1;
    }
    best
}

fn column_min(column: usize) -> i32 {
    let mut best = COST_MATRIX[column];
    let mut row = 1;
    while row < DIM {
        let value = COST_MATRIX[row * DIM + column];
        if value < best {
            best = value;
        }
        row += 1;
    }
    best
}

fn reduced_cost(row: usize, column: usize) -> i32 {
    COST_MATRIX[row * DIM + column] - row_min(row) - column_min(column)
}

#[unsafe(no_mangle)]
pub extern "C" fn hungarian_support_checksum() -> i32 {
    let mut checksum = 0i32;
    let mut row = 0usize;
    while row < DIM {
        checksum = checksum.wrapping_add(row_min(row).wrapping_mul(row as i32 + 1));
        row += 1;
    }

    let mut column = 0usize;
    while column < DIM {
        checksum = checksum.wrapping_add(column_min(column).wrapping_mul(column as i32 + 3));
        column += 1;
    }

    checksum
        .wrapping_add(reduced_cost(1, 2))
        .wrapping_add(reduced_cost(3, 0))
}
