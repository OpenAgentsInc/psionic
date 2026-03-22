use psionic_runtime::{
    tassadar_post_article_plugin_example_words_runtime_bundle_path,
    write_example_words_runtime_bundle,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = tassadar_post_article_plugin_example_words_runtime_bundle_path();
    let bundle = write_example_words_runtime_bundle(&output_path)?;
    println!("wrote {} ({})", output_path.display(), bundle.bundle_digest);
    Ok(())
}
