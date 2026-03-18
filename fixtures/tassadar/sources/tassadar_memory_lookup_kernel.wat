(module
  (memory 1)
  (data (i32.const 0) "\0b\00\00\00\13\00\00\00\17\00\00\00")
  (func (export "load_middle") (result i32)
    i32.const 4
    i32.load)
  (func (export "load_edge_sum") (result i32)
    i32.const 0
    i32.load
    i32.const 8
    i32.load
    i32.add))
