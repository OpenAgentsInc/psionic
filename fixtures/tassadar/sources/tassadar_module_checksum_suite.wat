(module
  (memory 1)
  (data (i32.const 0) "\04\00\00\00\05\00\00\00\06\00\00\00\07\00\00\00")
  (func (export "checksum_sum") (result i32)
    i32.const 0
    i32.load
    i32.const 4
    i32.load
    i32.add
    i32.const 8
    i32.load
    i32.add
    i32.const 12
    i32.load
    i32.add)
  (func (export "checksum_weighted") (result i32)
    i32.const 0
    i32.load
    i32.const 2
    i32.mul
    i32.const 4
    i32.load
    i32.add
    i32.const 8
    i32.load
    i32.add))
