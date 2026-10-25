use std::process::ExitCode;

use trine::cli;
use trine::convert;

fn main() -> ExitCode {
    let options = match cli::parse() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(error.exit_code() as u8);
        }
    };
    match convert::run(&options) {
        Ok(summary) => {
            if !options.quiet {
                let outputs: Vec<String> = summary
                    .outputs
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect();
                println!(
                    "Converted {} -> {} ({} triangles, {} vertices)",
                    summary.input.display(),
                    outputs.join(", "),
                    summary.triangles,
                    summary.vertices
                );
                if options.verbose {
                    eprintln!(
                        "materials: {}  images: {}  warnings: {}",
                        summary.materials, summary.images, summary.warning_count
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(error.exit_code() as u8)
        }
    }
}
