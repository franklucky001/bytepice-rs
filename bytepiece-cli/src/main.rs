use std::fmt::Display;
use clap::Parser;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use bytepiece_rs::TrainerBuilder;


fn vocab_sizes_parser(s:& str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|s| s.trim().parse().map_err(|_|
            format!("Invalid number in array: {}", s)
        ))
        .collect()
}
/// BytePiece training arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct TrainingArgs {
    /// Order of the model (default: 6)
    #[arg(long, default_value = "6")]
    order: u8,

    /// Maximum vocabulary size (ignored if max-vocab-size-array is set)
    #[arg(long, default_value = "10000")]
    max_vocab_size: usize,

    /// Array of vocabulary sizes (comma-separated list, e.g., "8000,16000,32000")
    #[arg(long, value_parser=vocab_sizes_parser)]
    max_vocab_size_array: Option<Vec<usize>>,

    /// Maximum piece length (default: 36)
    #[arg(long, default_value = "36")]
    max_piece_len: usize,

    /// Minimum count for a piece (default: 2)
    #[arg(long, default_value = "2")]
    min_count: usize,

    /// Maximum norm length (default: 10000)
    #[arg(long, default_value = "10000")]
    max_norm_len: usize,

    /// Whether to isolate digits (default: false)
    #[arg(long, default_value = "false")]
    isolate_digits: bool,

    /// Ensure Unicode validity (default: true)
    #[arg(long, default_value = "true")]
    ensure_unicode: bool,

    /// max workers for parallel training if value greater than 1
    #[arg(long, default_value = "1")]
    workers: usize,

    /// batch size for parallel training if workers > 1
    #[arg(long, default_value = "100")]
    batch_size: usize,

    #[arg(short, long, required = true)]
    input_file: String,

    #[arg(short, long, required = true)]
    output_file: String,
}

impl Display for TrainingArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.workers > 1{
            writeln!(f, "Training in parallel mode")?;
            writeln!(f, "  workers: {}", self.workers)?;
            writeln!(f, "  batch-size: {}", self.batch_size)?;
        }
        writeln!(f, "Training arguments detail")?;
        writeln!(f, "  order: {}", self.order)?;
        match &self.max_vocab_size_array {
            Some(array) => {
                writeln!(f, "  max-vocab-size: {:?}", array)?;
            },
            None => {
                writeln!(f, "  max-vocab-size: {}", self.max_vocab_size)?;
            }
        }
        writeln!(f, "  max-piece-len: {}", self.max_vocab_size)?;
        writeln!(f, "  min-count: {}", self.min_count)?;
        writeln!(f, "  max-norm-len: {}", self.max_norm_len)?;
        writeln!(f, "  isolate-digits: {}", self.isolate_digits)?;
        writeln!(f, "  ensure-unicode: {}", self.ensure_unicode)?;
        writeln!(f, "File information")?;
        writeln!(f, "  input-file: {}", self.input_file)?;
        writeln!(f, " output-file: {}", self.output_file)
    }
}
fn read_samples(path: & str) -> io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut texts = Vec::new();

    for line in reader.lines() {
        let line = line?;
        texts.push(line);
    }
    Ok(texts)
}

fn training(training_args: TrainingArgs) {
    let builder  = TrainerBuilder::new()
        .order(training_args.order)
        .max_piece_len(training_args.max_piece_len)
        .min_count(training_args.min_count)
        .max_norm_len(training_args.max_norm_len)
        .isolate_digits(training_args.isolate_digits)
        .ensure_unicode(training_args.ensure_unicode);
    let mut trainer = match training_args.max_vocab_size_array {
        Some(vocab_size_array) => {
            builder.max_vocab_size_array(vocab_size_array).build()
        },
        None => {
            builder.max_vocab_size(training_args.max_vocab_size).build()
        }
    };
    let samples = read_samples(&training_args.input_file).expect("read samples error");
    if training_args.workers == 1 {
        trainer.train(&samples);
    }else {
        trainer.parallel_train(&samples, training_args.batch_size, training_args.workers);
    }
    trainer.save(&training_args.output_file).expect(&format!("Error saving training file: {}", training_args.output_file));
}

fn main() {
    let args = TrainingArgs::parse();
    println!("Starting training...");
    println!("{}", args);
    training(args);
    println!("training done!");
}
