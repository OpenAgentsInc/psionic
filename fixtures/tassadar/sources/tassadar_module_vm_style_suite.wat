(module
  (memory 1)
  (data (i32.const 0) "\01\00\00\00\05\00\00\00\06\00\00\00\02\00\00\00\03\00\00\00\04\00\00\00")
  (func (export "dispatch_add") (result i32)
    i32.const 4
    i32.load
    i32.const 8
    i32.load
    i32.add)
  (func (export "dispatch_mul_add") (result i32)
    i32.const 12
    i32.load
    i32.const 16
    i32.load
    i32.mul
    i32.const 20
    i32.load
    i32.add))
