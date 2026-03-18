(module
  (memory 1)
  (data (i32.const 0) "\03\00\00\00\01\00\00\00\04\00\00\00")
  (func (export "parse_triplet") (result i32)
    i32.const 0
    i32.load
    i32.const 100
    i32.mul
    i32.const 4
    i32.load
    i32.const 10
    i32.mul
    i32.add
    i32.const 8
    i32.load
    i32.add)
  (func (export "parse_gap") (result i32)
    i32.const 8
    i32.load
    i32.const 0
    i32.load
    i32.sub))
