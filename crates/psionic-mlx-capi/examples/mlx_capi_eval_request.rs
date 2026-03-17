use psionic_mlx_capi::{compatibility_scope_json_string, eval_json_string};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scope: serde_json::Value = serde_json::from_str(&compatibility_scope_json_string())?;
    assert_eq!(scope["schema_version"], 1);

    let response = eval_json_string(
        r#"{
            "backend": "cpu",
            "seed": 7,
            "steps": [
                {"id": "lhs", "op": "ones", "shape": [2, 2]},
                {"id": "rhs", "op": "full", "shape": [2, 2], "value": 2.0},
                {"id": "sum", "op": "add", "lhs": "lhs", "rhs": "rhs"},
                {"id": "reduced", "op": "sum_axis", "input": "sum", "axis": 1}
            ],
            "output": "reduced"
        }"#,
    );
    let response: serde_json::Value = serde_json::from_str(&response)?;
    assert_eq!(response["status"], "ok");
    assert_eq!(response["shape"], serde_json::json!([2]));

    Ok(())
}
