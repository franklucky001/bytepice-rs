use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs::File;
use std::thread;
use std::time::Duration;
use anyhow::{self, Context};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use ndarray::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::bp_normalize::*;
use crate::bp_ngram::NGramProcessor;
use crate::bp_schema::{PieceDecoder, PieceMap};
use crate::bp_tokenizer::Tokenizer;

pub struct Trainer{
    order: u8,
    max_vocab_size_array: Vec<usize>,
    max_piece_len: usize,
    min_count: usize,
    // isolate_digits: bool,
    ensure_unicode: bool,
    ngrams: Vec<HashMap<Vec<u8>, f64>>,
    pieces: Vec<HashMap<Vec<u8>, usize>>,
    trans: Array2<f64>,
    normalizer: TextNormalizer,
    ngram_processor: NGramProcessor
}
impl Trainer {
    fn new(
        order: u8,
        max_vocab_size_array: Vec<usize>,
        max_piece_len: usize,
        min_count: usize,
        // isolate_digits: bool,
        ensure_unicode: bool,
        normalizer: TextNormalizer,
        ngram_processor: NGramProcessor
    ) -> Self {
        Self{
            order,
            max_vocab_size_array,
            max_piece_len,
            min_count,
            // isolate_digits,
            ensure_unicode,
            ngrams: Vec::new(),
            pieces: Vec::new(),
            trans: Self::build_trans(order as usize),
            normalizer,
            ngram_processor
        }
    }
    // 构造转移矩阵
    fn build_trans(order: usize) -> Array2<f64> {
        let mut trans = Array2::from_elem((order, order), f64::NEG_INFINITY);

        for i in 0..order {
            trans[[i, 0]] = 0.0;
            trans[[i, (i + 1).min(order - 1)]] = 0.0;
        }

        trans
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let paths = if self.pieces.len() == 1 {
            vec![path.to_string()]
        } else {
            // 为每个vocab size创建路径
            self.max_vocab_size_array
                .iter()
                .map(|size| format!("{}.{}", path, size))
                .collect::<Vec<_>>()
        };
        // 遍历pieces和paths
        for (pieces, path) in self.pieces.iter().zip(paths.iter()) {
            // 创建PieceMap并获取inner HashMap
            let piece_mapper = PieceMap::from(pieces);
            let piece_states = piece_mapper
                .inner()
                .into_iter()
                .map(|(k, v)|(k.encode(), v))
                .collect::<HashMap<_, _>>();
            // 序列化为JSON并写入文件
            // 创建文件并直接写入JSON
            let file = File::create(path).with_context(|| "create file failed")?;
            serde_json::to_writer_pretty(file, &piece_states).with_context(|| "serialize to json failed")?;
        }
        Ok(())
    }

    pub fn train<'a, S, I>(&mut self, texts: I)
    where S:'a + AsRef<str>,
          I: IntoIterator<Item=&'a S>
    {
        let norm_texts: Vec<Vec<u8>> = texts
            .into_iter()
            .map(|text|self.normalizer.normalize(text.as_ref()))
            .flatten()
            .collect();
        let ngrams = self.ngram_processor.count_ngrams(&norm_texts);
        let prune_ngrams = self.ngram_processor.prune_ngrams(ngrams);
        self.ngrams.extend(prune_ngrams.into_iter());
        let pieces = self.count_pieces(&norm_texts);
        let prune_pieces = self.prune_pieces(pieces);
        self.pieces.extend(prune_pieces.into_iter());
    }
    fn set_rayon_threads(workers: usize){
        ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .expect("Global thread pool setting failed.");
    }
    pub fn parallel_train<'a, S, I>(&mut self, texts: I, batch_size: usize, workers: usize)
    where S: 'a + AsRef<str>,
          I: IntoIterator<Item=&'a S>
    {
        Self::set_rayon_threads(workers);
        let norm_texts: Vec<Vec<u8>> = texts
            .into_iter()
            .map(|text|self.normalizer.normalize(text.as_ref()))
            .flatten()
            .collect();
        let ngrams = self.ngram_processor.parallel_count_ngrams(&norm_texts, batch_size);
        let prune_ngrams = self.ngram_processor.prune_ngrams(ngrams);
        self.ngrams.extend(prune_ngrams.into_iter());
        let pieces = self.parallel_count_pieces(&norm_texts, batch_size);
        let prune_pieces = self.parallel_prune_pieces(pieces, batch_size);
        self.pieces.extend(prune_pieces.into_iter())
    }

    fn count_pieces<'a, T>(&self, texts:&'a T) -> HashMap<&'a [u8], usize>
    where
        T: AsRef<[Vec<u8>]> + ?Sized,
    {
        let mut pieces = HashMap::new();
        // 创建进度条并设置样式
        let pb = ProgressBar::new(texts.as_ref().len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {elapsed_precise} {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("- ")
        );
        // 设置开始消息
        println!("Counting Pieces...");
        // 遍历所有文本
        for text in texts.as_ref() {
            // 对每个文本进行tokenize
            let tokens = self._tokenize(text);
            // 计数每个piece
            for token in tokens {
                *pieces.entry(&token[..]).or_insert(0) += 1;
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
        println!("Counting Pieces Done");
        pieces
    }
    fn _count_pieces<'a, T>(&self, texts:&'a T) -> HashMap<&'a [u8], usize>
    where
        T: AsRef<[Vec<u8>]> + ?Sized,
    {
        let mut pieces = HashMap::new();
        // 遍历所有文本
        for text in texts.as_ref() {
            // 对每个文本进行tokenize
            let tokens = self._tokenize(text);
            // 计数每个piece
            for token in tokens {
                *pieces.entry(&token[..]).or_insert(0) += 1;
            }
        }
        pieces
    }
    fn parallel_count_pieces<'a>(
        &mut self,
        texts: &'a Vec<Vec<u8>>,
        batch_size: usize
    ) -> HashMap<&'a [u8], usize>
    {
        // 创建进度条并设置样式
        let pb = ProgressBar::new(texts.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {elapsed_precise} {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("- ")
        );
        // 设置开始消息
        println!("Counting Pieces parallel...");
        let result = texts
            .par_chunks(batch_size)
            .map(|chunk| {
                pb.inc(chunk.len() as u64);
                thread::sleep(Duration::from_millis(2500));
                self._count_pieces(chunk)
            })
            .reduce(
                || HashMap::new(),  // 初始值为一个空HashMap
                |mut acc, cur| {
                    // 合并所有的HashMap
                    for (k, v) in cur.into_iter(){
                        *acc.entry(k).or_insert(0) += v;
                    }
                    acc
                }
            );
        pb.finish_and_clear();
        println!("Counting Pieces parallel Done");
        result
    }

    fn prune_pieces(&self, pieces: HashMap<&[u8], usize>) -> Vec<HashMap<Vec<u8>, usize>> {
        // 首先转换为 owned 类型
        let mut owned_pieces = pieces
            .into_iter()
            .map(|(k, v)| {
                (k.to_vec(), v)
            })
            .collect::<HashMap<_, _>>();

        // 处理单字节情况
        for i in 0..=255u8 {
            let byte = vec![i];
            if !owned_pieces.contains_key(&byte) {
                owned_pieces.insert(byte, 1);
            }
        }
        // 按频率和长度进行剪枝
        let (mut keep_pieces, drop_pieces): (HashMap<Vec<u8>, usize>, HashMap<Vec<u8>, usize>) = owned_pieces
            .into_iter()
            .partition(|(k, v)| {
                k.len() == 1 || (k.len() <= self.max_piece_len && *v >= self.min_count)
            });
        let split_pieces = self.split_pieces(&keep_pieces, drop_pieces);
        for (k, v) in split_pieces {
            keep_pieces.entry(k).and_modify(|x| *x += v);
        }
        // 循环剪枝浪费的pieces
        loop {
            let len_keep_pieces = keep_pieces.len();

            // 创建新的drop_pieces用于下一轮剪枝
            let drop_pieces: HashMap<_, _> = keep_pieces.clone();

            // 进行split_pieces操作
            let split_result = self.split_pieces(&keep_pieces, drop_pieces);

            // 更新keep_pieces
            keep_pieces = split_result;

            // 如果长度没有变化，说明达到稳定状态，退出循环
            if len_keep_pieces == keep_pieces.len() {
                break;
            }
            // 可以添加进度显示（如果需要）
        }
        let mut final_pieces = Vec::new();
        let mut keep_len = keep_pieces.len();
        for max_vocab_size in self.max_vocab_size_array.iter() {
            if keep_len + 3 <= *max_vocab_size{
                final_pieces.push(keep_pieces.clone());
                continue;
            }
            // 按条件排序pieces
            let mut pieces: Vec<_> = keep_pieces.clone().into_iter().collect();
            pieces.sort_by(|(k1, v1), (k2, v2)| {
                // 排序条件: (len > 1, -freq, -len, bytes)
                let c1 = (k1.len() > 1, Reverse(v1), Reverse(k1.len()), k1);
                let c2 = (k2.len() > 1, Reverse(v2), Reverse(k2.len()), k2);
                c1.cmp(&c2)  // 降序排列
            });

            // 分割keep_pieces和drop_pieces
            let (keep_part, drop_part) = pieces.split_at(max_vocab_size - 3);

            // 转换回HashMap
            let mut keep_pieces: HashMap<_, _> = keep_part
                .into_iter()
                .map(|(k, v)| (k.to_owned(), *v))
                .collect();

            // 处理drop_pieces
            let drop_pieces: HashMap<_, _> = drop_part.iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();

            // 更新keep_pieces
            let split_pieces = self.split_pieces(&keep_pieces, drop_pieces);
            for (k, v) in split_pieces {
                keep_pieces.entry(k).and_modify(|x| *x += v);
            }

            // 循环剪枝浪费的pieces
            loop {
                let len_keep_pieces = keep_pieces.len();

                // 创建进度条（如果使用indicatif）

                let drop_pieces = keep_pieces.clone();
                let split_result = self.split_pieces(&keep_pieces, drop_pieces);
                keep_pieces = split_result;
                if len_keep_pieces == keep_pieces.len() {
                    break;
                }
            }
            keep_len = keep_pieces.len();
            final_pieces.push(keep_pieces);
        }
        final_pieces
    }

    fn parallel_prune_pieces(&self, pieces: HashMap<&[u8], usize>, batch_size: usize) -> Vec<HashMap<Vec<u8>, usize>> {
        // 首先转换为 owned 类型
        let mut owned_pieces = pieces
            .into_iter()
            .map(|(k, v)| {
                (k.to_vec(), v)
            })
            .collect::<HashMap<_, _>>();

        // 处理单字节情况
        for i in 0..=255u8 {
            let byte = vec![i];
            if !owned_pieces.contains_key(&byte) {
                owned_pieces.insert(byte, 1);
            }
        }
        // 按频率和长度进行剪枝
        let (mut keep_pieces, drop_pieces): (HashMap<Vec<u8>, usize>, HashMap<Vec<u8>, usize>) = owned_pieces
            .into_iter()
            .partition(|(k, v)| {
                k.len() == 1 || (k.len() <= self.max_piece_len && *v >= self.min_count)
            });
        let split_pieces = self.parallel_split_pieces(&keep_pieces, drop_pieces, batch_size);
        for (k, v) in split_pieces {
            keep_pieces.entry(k).and_modify(|x| *x += v);
        }
        // 循环剪枝浪费的pieces
        loop {
            let len_keep_pieces = keep_pieces.len();

            // 创建新的drop_pieces用于下一轮剪枝
            let drop_pieces: HashMap<_, _> = keep_pieces.clone();

            // 进行split_pieces操作
            let split_result = self.parallel_split_pieces(&keep_pieces, drop_pieces, batch_size);

            // 更新keep_pieces
            keep_pieces = split_result;

            // 如果长度没有变化，说明达到稳定状态，退出循环
            if len_keep_pieces == keep_pieces.len() {
                break;
            }
            // 可以添加进度显示（如果需要）
        }
        let mut final_pieces = Vec::new();
        let mut keep_len = keep_pieces.len();
        for max_vocab_size in self.max_vocab_size_array.iter() {
            if keep_len + 3 <= *max_vocab_size{
                final_pieces.push(keep_pieces.clone());
                continue;
            }
            // 按条件排序pieces
            let mut pieces: Vec<_> = keep_pieces.clone().into_iter().collect();
            pieces.sort_by(|(k1, v1), (k2, v2)| {
                // 排序条件: (len > 1, -freq, -len, bytes)
                let c1 = (k1.len() > 1, Reverse(v1), Reverse(k1.len()), k1);
                let c2 = (k2.len() > 1, Reverse(v2), Reverse(k2.len()), k2);
                c1.cmp(&c2)  // 降序排列
            });

            // 分割keep_pieces和drop_pieces
            let (keep_part, drop_part) = pieces.split_at(max_vocab_size - 3);

            // 转换回HashMap
            let mut keep_pieces: HashMap<_, _> = keep_part
                .into_iter()
                .map(|(k, v)| (k.to_owned(), *v))
                .collect();

            // 处理drop_pieces
            let drop_pieces: HashMap<_, _> = drop_part.iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();

            // 更新keep_pieces
            let split_pieces = self.parallel_split_pieces(&keep_pieces, drop_pieces, batch_size);
            for (k, v) in split_pieces {
                keep_pieces.entry(k).and_modify(|x| *x += v);
            }

            // 循环剪枝浪费的pieces
            loop {
                let len_keep_pieces = keep_pieces.len();

                // 创建进度条（如果使用indicatif）

                let drop_pieces = keep_pieces.clone();
                let split_result = self.parallel_split_pieces(&keep_pieces, drop_pieces, batch_size);
                keep_pieces = split_result;
                if len_keep_pieces == keep_pieces.len() {
                    break;
                }
            }
            keep_len = keep_pieces.len();
            final_pieces.push(keep_pieces);
        }
        final_pieces
    }

    fn split_pieces(&self, keep_pieces:& HashMap<Vec<u8>, usize>, drop_pieces: HashMap<Vec<u8>, usize>) -> HashMap<Vec<u8>, usize>
    {
        let piece_mapper = PieceMap::from(keep_pieces);
        let tokenizer = Tokenizer::new(piece_mapper.inner(), None);
        drop_pieces
            .iter()
            .flat_map(|(k, v)|{
                tokenizer._tokenize(k, 1.0).into_iter().map(move|piece|(piece, v))
            })
            .fold(
                HashMap::new(),
                |mut counter, (piece, v)| {
                    *counter.entry(piece).or_insert(0) += v;
                    counter
                }
            )
    }

    fn parallel_split_pieces(&self, keep_pieces:& HashMap<Vec<u8>, usize>, drop_pieces: HashMap<Vec<u8>, usize>, batch_size: usize) -> HashMap<Vec<u8>, usize>{
        let piece_mapper = PieceMap::from(keep_pieces);
        let tokenizer = Tokenizer::new(piece_mapper.inner(), None);
        drop_pieces
            .into_iter()
            .collect::<Vec<_>>()
            .par_chunks(batch_size)
            .map(|chunk|{
                // 处理每个batch
                chunk
                    .into_iter()
                    .flat_map(|(k, v)|{
                        tokenizer._tokenize(k, 1.0).into_iter().map(move|piece|(piece, v))
                    })
                    .fold(
                        HashMap::new(),
                        |mut counter, (piece, v)| {
                            *counter.entry(piece).or_insert(0) += v;
                            counter
                        }
                    )
            })
            .reduce(
                || HashMap::new(),
                |mut acc, cur| {
                    // 合并所有batch的结果
                    for (piece, v) in cur {
                        *acc.entry(piece).or_insert(0) += v;
                    }
                    acc
                }
            )
    }
    fn _tokenize<'a>(&self, text: &'a Vec<u8>) -> Vec<&'a [u8]> {
        let text_len = text.len();
        let order = self.order as usize;

        // 初始化nodes矩阵
        let mut nodes = Array2::from_elem((text_len, order), f64::NEG_INFINITY);

        // 填充nodes矩阵
        for j in 0..order {
            for i in j..text_len{
                if let Some(slice) = text.get(i.saturating_sub(j)..=i) {
                    if let Some(&score) = self.ngrams[j + 1].get(slice) {
                        nodes[[i, j]] = score;
                    }
                }
            }
        }

        // 处理Unicode字符（如果需要）
        if self.ensure_unicode {
            for (i, &byte) in text.iter().enumerate() {
                if byte >= 128 && byte < 192 {
                    nodes[[i, 0]] -= f64::INFINITY;
                }
            }
        }

        // Viterbi算法
        let mut routes = Array2::zeros((text_len - 1, order));
        for i in 1..text_len {
            let prev_scores = nodes.slice(s![i-1, ..]);
            let curr_scores = nodes.slice(s![i, ..]);

            // 计算scores矩阵
            let mut scores = Array2::zeros((order, order));
            for j in 0..order {
                for k in 0..order {
                    scores[[j, k]] = prev_scores[j] + self.trans[[j, k]] + curr_scores[k];
                }
            }

            // 更新routes和nodes
            for k in 0..order {
                let (max_idx, &max_val) = scores
                    .column(k)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                routes[[i-1, k]] = max_idx as i32;
                nodes[[i, k]] = max_val;
            }
        }

        // 构建最优路径
        let mut opt_route = Vec::new();
        let mut current = nodes
            .row(text_len - 1)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        opt_route.push(current);

        for i in (1..text_len).rev() {
            current = routes[[i-1, current]] as usize;
            opt_route.push(current);
        }
        opt_route.reverse();

        // 找到所有0的位置
        let mut boundaries = opt_route
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r == 0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        boundaries.push(text_len);

        // 构建结果
        boundaries
            .windows(2)
            .map(|w| &text[w[0]..w[1]])
            .collect()
    }
}
enum MaxVocabSize{
    VocabSize(usize),
    VocabSizeArray(Vec<usize>),
}
pub struct TrainerBuilder {
    order: u8,
    max_vocab_size: MaxVocabSize,
    max_piece_len: usize,
    min_count: usize,
    max_norm_len: usize,
    isolate_digits: bool,
    ensure_unicode: bool,
}

impl TrainerBuilder {
    pub fn new() -> Self {
        Self{
            order: 6,
            max_vocab_size: MaxVocabSize::VocabSize(10000),
            max_piece_len: 36,
            min_count: 2,
            max_norm_len: 10000,
            isolate_digits: false,
            ensure_unicode: true
        }
    }
    pub fn order(mut self, order: u8) -> Self {
        self.order = order;
        self
    }
    pub fn max_vocab_size(mut self, max_vocab_size: usize) -> Self {
        self.max_vocab_size = MaxVocabSize::VocabSize(max_vocab_size);
        self
    }
    pub fn max_vocab_size_array(mut self, mut max_vocab_size_array: Vec<usize>) -> Self {
        max_vocab_size_array.sort_by(|a, b| b.cmp(a));
        self.max_vocab_size = MaxVocabSize::VocabSizeArray(max_vocab_size_array);
        self
    }
    pub fn max_piece_len(mut self, max_piece_len: usize) -> Self {
        self.max_piece_len = max_piece_len;
        self
    }
    pub fn min_count(mut self, min: usize) -> Self {
        self.min_count = min;
        self
    }
    pub fn max_norm_len(mut self, max_norm_len: usize) -> Self {
        self.max_norm_len = max_norm_len;
        self
    }
    pub fn isolate_digits(mut self, isolate_digits: bool) -> Self {
        self.isolate_digits = isolate_digits;
        self
    }
    pub fn ensure_unicode(mut self, ensure_unicode: bool) -> Self {
        self.ensure_unicode = ensure_unicode;
        self
    }
    pub fn build(self) -> Trainer {
        let max_vocab_size_array = match self.max_vocab_size {
            MaxVocabSize::VocabSize(max_vocab_size) => vec![max_vocab_size],
            MaxVocabSize::VocabSizeArray(max_vocab_size_array) => max_vocab_size_array,
        };
        let normalizer = TextNormalizer::new(self.max_norm_len, self.isolate_digits);
        let ngram_processor = NGramProcessor::new(self.order, self.min_count);
        Trainer::new(self.order, max_vocab_size_array, self.max_piece_len, self.min_count,  self.ensure_unicode, normalizer, ngram_processor)
    }
}

