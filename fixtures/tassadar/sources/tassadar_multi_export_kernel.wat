(module
  (func (export "pair_sum") (result i32)
    i32.const 2
    i32.const 3
    i32.add)
  (func (export "local_double") (result i32)
    (local i32)
    i32.const 7
    local.set 0
    local.get 0
    local.get 0
    i32.add))
