#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

const DIM: usize = 10;
const BEST_COST_INIT: i32 = 1_000_000;
const COST_MATRIX: [i32; DIM * DIM] = [
    61, 58, 35, 86, 32, 39, 41, 27, 21, 42, //
    59, 77, 97, 99, 78, 21, 89, 72, 35, 63, //
    88, 85, 37, 57, 59, 97, 37, 29, 69, 94, //
    32, 82, 53, 20, 77, 96, 21, 70, 50, 61, //
    15, 44, 81, 10, 64, 36, 56, 78, 20, 69, //
    76, 35, 87, 69, 16, 55, 26, 37, 30, 66, //
    86, 32, 74, 94, 32, 14, 24, 12, 31, 70, //
    97, 63, 20, 64, 90, 21, 28, 49, 89, 10, //
    58, 52, 27, 76, 61, 35, 17, 91, 37, 66, //
    42, 79, 61, 26, 55, 98, 70, 17, 26, 86,
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

fn build_remaining_bounds() -> [i32; DIM + 1] {
    let mut bounds = [0; DIM + 1];
    let mut index = DIM;
    while index > 0 {
        let next = index - 1;
        bounds[next] = bounds[index] + row_min(next);
        index -= 1;
    }
    bounds
}

fn search(
    row: usize,
    current_cost: i32,
    used_columns: &mut [bool; DIM],
    remaining_bounds: &[i32; DIM + 1],
    best_cost: &mut i32,
) {
    if row == DIM {
        if current_cost < *best_cost {
            *best_cost = current_cost;
        }
        return;
    }
    if current_cost + remaining_bounds[row] >= *best_cost {
        return;
    }

    let mut column = 0;
    while column < DIM {
        if !used_columns[column] {
            used_columns[column] = true;
            search(
                row + 1,
                current_cost + COST_MATRIX[row * DIM + column],
                used_columns,
                remaining_bounds,
                best_cost,
            );
            used_columns[column] = false;
        }
        column += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hungarian_10x10_article_cost() -> i32 {
    let remaining_bounds = build_remaining_bounds();
    let mut used_columns = [false; DIM];
    let mut best_cost = BEST_COST_INIT;
    search(
        0,
        0,
        &mut used_columns,
        &remaining_bounds,
        &mut best_cost,
    );
    best_cost
}
