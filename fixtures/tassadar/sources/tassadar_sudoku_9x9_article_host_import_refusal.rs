#![no_std]
#![no_main]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

unsafe extern "C" {
    fn imported_sudoku_seed(index: i32) -> i32;
}

#[unsafe(no_mangle)]
pub extern "C" fn sudoku_9x9_article_host_import_checksum() -> i32 {
    let mut checksum = 0;
    let mut index = 0;
    while index < 9 {
        checksum += unsafe { imported_sudoku_seed(index) } * (index + 1);
        index += 1;
    }
    checksum
}
