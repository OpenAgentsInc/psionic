#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const GRID_WIDTH: usize = 9;
const CELL_COUNT: usize = GRID_WIDTH * GRID_WIDTH;
const MASKED_INDICES: [usize; 27] = [
    0, 2, 5, 8, 10, 13, 16, 20, 24, 28, 31, 34, 36, 40, 44, 46, 49, 52, 56, 60, 64, 67, 70, 72,
    75, 78, 80,
];
const SOLVED_GRID: [i32; CELL_COUNT] = [
    5, 3, 4, 6, 7, 8, 9, 1, 2, //
    6, 7, 2, 1, 9, 5, 3, 4, 8, //
    1, 9, 8, 3, 4, 2, 5, 6, 7, //
    8, 5, 9, 7, 6, 1, 4, 2, 3, //
    4, 2, 6, 8, 5, 3, 7, 9, 1, //
    7, 1, 3, 9, 2, 4, 8, 5, 6, //
    9, 6, 1, 5, 3, 7, 2, 8, 4, //
    2, 8, 7, 4, 1, 9, 6, 3, 5, //
    3, 4, 5, 2, 8, 6, 1, 7, 9,
];

fn build_puzzle() -> [i32; CELL_COUNT] {
    let mut puzzle = SOLVED_GRID;
    let mut index = 0;
    while index < MASKED_INDICES.len() {
        puzzle[MASKED_INDICES[index]] = 0;
        index += 1;
    }
    puzzle
}

fn row_is_valid(grid: &[i32; CELL_COUNT], row: usize, value: i32) -> bool {
    let mut column = 0;
    while column < GRID_WIDTH {
        if grid[row * GRID_WIDTH + column] == value {
            return false;
        }
        column += 1;
    }
    true
}

fn column_is_valid(grid: &[i32; CELL_COUNT], column: usize, value: i32) -> bool {
    let mut row = 0;
    while row < GRID_WIDTH {
        if grid[row * GRID_WIDTH + column] == value {
            return false;
        }
        row += 1;
    }
    true
}

fn box_is_valid(grid: &[i32; CELL_COUNT], row: usize, column: usize, value: i32) -> bool {
    let start_row = (row / 3) * 3;
    let start_column = (column / 3) * 3;
    let mut box_row = 0;
    while box_row < 3 {
        let mut box_column = 0;
        while box_column < 3 {
            let index = (start_row + box_row) * GRID_WIDTH + start_column + box_column;
            if grid[index] == value {
                return false;
            }
            box_column += 1;
        }
        box_row += 1;
    }
    true
}

fn next_empty(grid: &[i32; CELL_COUNT]) -> Option<usize> {
    let mut index = 0;
    while index < CELL_COUNT {
        if grid[index] == 0 {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn solve(grid: &mut [i32; CELL_COUNT]) -> bool {
    let Some(index) = next_empty(grid) else {
        return true;
    };
    let row = index / GRID_WIDTH;
    let column = index % GRID_WIDTH;
    let mut candidate = 1;
    while candidate <= 9 {
        if row_is_valid(grid, row, candidate)
            && column_is_valid(grid, column, candidate)
            && box_is_valid(grid, row, column, candidate)
        {
            grid[index] = candidate;
            if solve(grid) {
                return true;
            }
            grid[index] = 0;
        }
        candidate += 1;
    }
    false
}

#[unsafe(no_mangle)]
pub extern "C" fn sudoku_9x9_article_checksum() -> i32 {
    let mut puzzle = build_puzzle();
    if !solve(&mut puzzle) {
        return -1;
    }
    let mut checksum = 0;
    let mut index = 0;
    while index < CELL_COUNT {
        checksum += puzzle[index] * (index as i32 + 1);
        index += 1;
    }
    checksum
}
