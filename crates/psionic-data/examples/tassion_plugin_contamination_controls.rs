use psionic_data::{
    tassion_plugin_contamination_bundle_path, write_tassion_plugin_contamination_bundle,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = tassion_plugin_contamination_bundle_path();
    let bundle = write_tassion_plugin_contamination_bundle(&output_path)?;
    println!(
        "wrote {} with digest {}",
        output_path.display(),
        bundle.bundle_digest
    );
    println!("dataset identity: {}", bundle.dataset_identity);
    println!("parent lineage rows: {}", bundle.parent_lineage_rows.len());
    Ok(())
}
