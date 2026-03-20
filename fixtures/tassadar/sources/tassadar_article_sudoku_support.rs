#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const GRID_WIDTH: usize = 9;
const CELL_COUNT: usize = GRID_WIDTH * GRID_WIDTH;
const PUZZLE: [i32; CELL_COUNT] = [
    0, 0, 4, 6, 7, 8, 9, 1, 2, //
    6, 7, 2, 1, 9, 5, 3, 4, 8, //
    1, 9, 8, 3, 4, 2, 5, 6, 7, //
    8, 5, 9, 7, 6, 1, 4, 2, 3, //
    4, 2, 6, 8, 0, 3, 7, 9, 1, //
    7, 1, 3, 9, 2, 4, 8, 5, 6, //
    9, 6, 1, 5, 3, 7, 2, 8, 4, //
    2, 8, 7, 4, 1, 9, 6, 0, 5, //
    3, 4, 5, 2, 8, 6, 1, 7, 0,
];

fn row_has_value(row: usize, value: i32) -> bool {
    let mut column = 0usize;
    while column < GRID_WIDTH {
        if PUZZLE[row * GRID_WIDTH + column] == value {
            return true;
        }
        column += 1;
    }
    false
}

fn column_has_value(column: usize, value: i32) -> bool {
    let mut row = 0usize;
    while row < GRID_WIDTH {
        if PUZZLE[row * GRID_WIDTH + column] == value {
            return true;
        }
        row += 1;
    }
    false
}

fn box_has_value(row: usize, column: usize, value: i32) -> bool {
    let start_row = (row / 3) * 3;
    let start_column = (column / 3) * 3;
    let mut local_row = 0usize;
    while local_row < 3 {
        let mut local_column = 0usize;
        while local_column < 3 {
            let index = (start_row + local_row) * GRID_WIDTH + start_column + local_column;
            if PUZZLE[index] == value {
                return true;
            }
            local_column += 1;
        }
        local_row += 1;
    }
    false
}

fn candidate_mask(index: usize) -> i32 {
    if PUZZLE[index] != 0 {
        return 0;
    }

    let row = index / GRID_WIDTH;
    let column = index % GRID_WIDTH;
    let mut value = 1i32;
    let mut mask = 0i32;
    while value <= 9 {
        if !row_has_value(row, value)
            && !column_has_value(column, value)
            && !box_has_value(row, column, value)
        {
            mask |= 1 << (value - 1);
        }
        value += 1;
    }
    mask
}

#[unsafe(no_mangle)]
pub extern "C" fn sudoku_support_checksum() -> i32 {
    let indices = [0usize, 1, 40, 70, 80];
    let mut checksum = 0i32;
    let mut index = 0usize;
    while index < indices.len() {
        checksum =
            checksum.wrapping_add(candidate_mask(indices[index]).wrapping_mul(index as i32 + 1));
        index += 1;
    }
    checksum
}
