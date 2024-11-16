use std::collections::HashMap;
use rayon::prelude::*;
use std::fs::File;
use std::hash::Hash;
use std::io::BufReader;
use crate::normalize;
use crate::bp_random::{RandomChooser, choice};
use crate::bp_automaton::{AutomatonBuilder, Automaton};
use crate::bp_schema::{PieceDecoder, PieceState};

pub struct Tokenizer {
    // // piece to value
    // p2v: HashMap<Vec<u8>, usize>,
    // // piece to id
    // p2i: HashMap<Vec<u8>, usize>,
    // // id to piece
    // i2p: HashMap<usize, Vec<u8>>,
    // vocab_size: usize,
    automaton: Automaton<Vec<u8>, (usize, f64)>,
}


impl Tokenizer{
    pub fn new<P>(piece_mapper: HashMap<P, PieceState>, seed: Option<u64>) -> Self
    where P: Hash + Eq + PieceDecoder
    {
        let mut p2v = HashMap::new();
        let mut p2i = HashMap::new();

        // 从piece_mapper构建映射，需要先解码base64
        for (encoded_piece, state) in piece_mapper {
            // base64解码
            let piece = encoded_piece.decode();

            p2v.insert(piece.clone(), state.value);
            p2i.insert(piece.clone(), state.pid);
        }

        // 初始化特殊token的id，转换为bytes
        let special_tokens = ["<pad>", "<bos>", "<eos>"];
        for (i, token) in special_tokens.iter().enumerate() {
            p2i.insert(token.as_bytes().to_vec(), i);
        }

        // 构建id到piece的反向映射
        // let i2p: HashMap<usize, Vec<u8>> = p2i.iter()
        //     .map(|(k, &v)| (v, k.clone()))
        //     .collect();
        // let vocab_size = p2i.len();
        let log_total: f64 = p2v.iter().map(|(_, v)| (*v as f64).ln()).sum();
        let mut builder = AutomatonBuilder::new();
        for (k, v) in p2v.iter(){
            let value = (k.len(), (*v as f64).ln() - log_total);
            builder.insert(k.to_vec(), value);
        }
        let automaton = builder.builder_automaton();
        RandomChooser::init_global_chooser(seed);
        Self{
            automaton
        }
    }
    pub fn from_file(filename: &str, seed: Option<u64>) -> Self{
        // 读取JSON文件
        let file = File::open(filename).expect("piece file open failed");
        let reader = BufReader::new(file);
        let mapper: HashMap<String, PieceState> = serde_json::from_reader(reader).expect("piece file read deserialize failed");
        Self::new(mapper, seed)
    }
    // log_sum_exp实现
    #[inline]
    fn log_sum_exp(x: f64, y: f64) -> f64 {
        if x < y {
            y + (1.0 + (x - y).exp()).ln()
        }else {
            x + (1.0 + (y - x).exp()).ln()
        }
    }

    pub(crate) fn _tokenize(& self, text: &[u8], alpha: f64) -> Vec<Vec<u8>> {
        let n = text.len();
        // 初始化scores和routes
        let mut scores = vec![0.0];
        scores.extend(vec![-f64::INFINITY; n]);
        let mut routes: Vec<usize> = (0..=n).collect();
        let mut tokens = Vec::new();

        // 遍历automaton的匹配结果
        for (end_idx, (k, v)) in self.automaton.iter(text) {
            let s = end_idx - *k;
            let e = end_idx;
            if alpha < 0.0 {
                // 贪婪模式
                let score = scores[s] + v;
                if score > scores[e] {
                    scores[e] = score;
                    routes[e] = s;
                }
            } else {
                // 随机采样模式
                let score = scores[s] + alpha * v;
                scores[e] = Self::log_sum_exp(scores[e], score);
                if choice(score, scores[e]) {
                    routes[e] = s;
                }
            }
        }

        // 回溯构建tokens
        let mut e = n;
        let mut text_slice = text;
        while !text_slice.is_empty() {
            let s = routes[e];
            tokens.push(text_slice[s..e].to_vec());
            text_slice = &text_slice[..s];
            e = s;
        }

        // 反转tokens并返回
        tokens.reverse();
        tokens
    }
    pub fn tokenize(&self, text: &str, alpha: f64) -> Vec<u8>{
        let pieces:Vec<Vec<u8>> = normalize!(text);
        // pieces.concat()
        pieces
            .into_iter()
            .flat_map(|piece| {self._tokenize(&piece, alpha)})
            .flatten()
            .collect::<Vec<_>>()
    }
    pub fn batch_tokenize<S>(&self, texts: Vec<S>, alpha: f64) -> Vec<Vec<u8>>
    where S: AsRef<str> + Send + Sync
    {
        texts
            .into_par_iter()
            .map(|text|self.tokenize(text.as_ref(), alpha))
            .collect()
    }
}