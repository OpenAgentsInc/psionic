(module
  (func (export "distance_tiny") (result i32)
    i32.const 2
    i32.const 3
    i32.add
  )
  (func (export "distance_small") (result i32)
    (local i32 i32)
    i32.const 2
    i32.const 3
    i32.add
    local.set 0
    i32.const 2
    i32.const 4
    i32.add
    local.set 1
    local.get 0
    local.get 1
    i32.add
  )
)
