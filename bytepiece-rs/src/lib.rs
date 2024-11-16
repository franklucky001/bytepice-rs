mod bp_trainer;
mod bp_tokenizer;
mod bp_ngram;
pub(crate) mod bp_normalize;
mod bp_automaton;
mod bp_random;
mod bp_schema;

pub use bp_trainer::{Trainer, TrainerBuilder};
pub use bp_tokenizer::Tokenizer;

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{self, BufRead, BufReader};
    use serde_json::Value;
    use crate::{Tokenizer, TrainerBuilder};

    #[test]
    fn test_tokenize() {
        let tokenizer = Tokenizer::from_file("./bytepiece.model", None);
        let result = tokenizer.tokenize("西游记是一部神话小说", -1.0);
        result.iter().for_each(|token| println!("\\x{:02x}", token));
    }
    fn read_text_from_jsonl(path: &str) -> io::Result<Vec<String>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut texts = Vec::new();

        for line in reader.lines() {
            let line = line?;
            // 解析JSON
            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                // 提取text字段
                if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
                    texts.push(text.to_string());
                }
            }
        }
        Ok(texts)
    }
    #[test]
    fn test_trainer(){
        let mut trainer = TrainerBuilder::new().build();
        let texts = read_text_from_jsonl("data/data_sample.json").expect("read sample data failed");
        // let samples = texts.iter().map(String::as_str).collect::<Vec<&str>>();
        trainer.train(texts);
        trainer.save("bytepiece_rs.model").expect("Unable to save trainer");
    }

    #[test]
    fn test_parallel_trainer(){
        let mut trainer = TrainerBuilder::new().build();
        let texts = read_text_from_jsonl("data/data_sample.json").expect("read sample data failed");
        // let samples = texts.iter().map(String::as_str).collect::<Vec<&str>>();
        trainer.parallel_train(texts, 2, 2);
        trainer.save("bytepiece_rs-multi.model").expect("Unable to save trainer");
    }
}

