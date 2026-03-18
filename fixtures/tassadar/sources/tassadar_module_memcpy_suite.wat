(module
  (memory 1)
  (data (i32.const 0) "\01\00\00\00\02\00\00\00\03\00\00\00")
  (func (export "copy_sum") (result i32)
    i32.const 16
    i32.const 0
    i32.load
    i32.store
    i32.const 20
    i32.const 4
    i32.load
    i32.store
    i32.const 24
    i32.const 8
    i32.load
    i32.store
    i32.const 16
    i32.load
    i32.const 20
    i32.load
    i32.add
    i32.const 24
    i32.load
    i32.add)
  (func (export "copy_tail") (result i32)
    i32.const 28
    i32.const 8
    i32.load
    i32.store
    i32.const 28
    i32.load))
