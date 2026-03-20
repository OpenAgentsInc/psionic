use std::vec::Vec;

const COST_MATRIX: [i32; 16] = [
    61, 58, 35, 86, //
    59, 77, 97, 99, //
    88, 85, 37, 57, //
    32, 82, 53, 20,
];

#[unsafe(no_mangle)]
pub extern "C" fn hungarian_10x10_article_std_refusal_cost() -> i32 {
    let mut values = Vec::from(COST_MATRIX);
    values.sort_unstable();
    values[0] + values[1] + values[2] + values[3]
}
