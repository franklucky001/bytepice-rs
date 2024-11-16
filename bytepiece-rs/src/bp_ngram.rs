use std::collections::HashMap;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::{thread, time::Duration};
pub(crate) struct NGramProcessor{
    order: u8,
    min_count: usize
}

impl NGramProcessor {
    pub fn new(order: u8, min_count: usize) -> Self {
        Self{
            order, min_count
        }
    }
    pub(crate) fn count_ngrams<'a, T>(& self, texts:&'a T) -> Vec<HashMap<&'a [u8], usize>>
    where
        T: AsRef<[Vec<u8>]> + ?Sized,
    {
        let mut ngrams = vec![HashMap::new(); (self.order + 1) as usize];
        // 创建进度条并设置样式
        let pb = ProgressBar::new(texts.as_ref().len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {elapsed_precise} {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("- ")
        );
        // 设置开始消息
        println!("Counting N-grams...");
        for text in texts.as_ref() {
            for i in 0..text.len() {
                for j in 0..=self.order as usize {
                    if i + j <= text.len() {
                        let k = &text[i..i + j];
                        *ngrams[j].entry(k).or_insert(0) += 1;
                    }
                }
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
        println!("Counting N-grams Done");
        ngrams
    }
    fn _count_ngrams<'a, T>(& self, texts:&'a T) -> Vec<HashMap<&'a [u8], usize>>
    where
        T: AsRef<[Vec<u8>]> + ?Sized,
    {
        let mut ngrams = vec![HashMap::new(); (self.order + 1) as usize];
        // 设置开始消息
        for text in texts.as_ref() {
            for i in 0..text.len() {
                for j in 0..=self.order as usize {
                    if i + j <= text.len() {
                        let k = &text[i..i + j];
                        *ngrams[j].entry(k).or_insert(0) += 1;
                    }
                }
            }
        }
        ngrams
    }
    // 并行计算n-grams
    pub(crate) fn parallel_count_ngrams<'a>(
        &self,
        texts: &'a Vec<Vec<u8>>,
        batch_size: usize
    ) -> Vec<HashMap<&'a [u8], usize>>
    {
        // 并行处理每个批次
        let order_plus_one = (self.order + 1) as usize;
        // 创建进度条并设置样式
        let pb = ProgressBar::new(texts.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {elapsed_precise} {bar:40.cyan/blue} {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("- ")
        );
        println!("Counting N-grams parallel...");
        // 使用reduce来直接合并结果，避免中间Vec的创建
        let result = texts.par_chunks(batch_size)
            .map(|batch|{
                pb.inc(batch.len() as u64);
                thread::sleep(Duration::from_millis(2500));
                self._count_ngrams(batch)
            })
            .reduce(
                // 初始值创建函数
                || vec![HashMap::new(); order_plus_one],
                // 合并函数
                |mut acc, curr| {
                    // 由于知道两个Vec的长度相同，可以直接用zip
                    for (acc_map, curr_map) in acc.iter_mut().zip(curr.iter()) {
                        for (k, &v) in curr_map {
                            *acc_map.entry(k).or_insert(0) += v;
                        }
                    }
                    acc
                }
            );
        pb.finish_and_clear();
        println!("Counting N-grams parallel Done");
        result
    }

    pub(crate) fn prune_ngrams(
        &self,
        ngrams: Vec<HashMap<&[u8], usize>>
    ) -> Vec<HashMap<Vec<u8>, f64>> {
        // 处理单字节情况
        // 首先转换为 owned 类型
        let mut owned_ngrams: Vec<HashMap<Vec<u8>, usize>> = ngrams
            .into_iter()
            .map(|map| {
                map.into_iter()
                    .map(|(k, v)| (k.to_vec(), v))
                    .collect()
            })
            .collect();

        // 处理单字节情况
        for i in 0..=255u8 {
            let byte = vec![i];
            if !owned_ngrams[1].contains_key(&byte) {
                owned_ngrams[1].insert(byte, 1);
                *owned_ngrams[0].entry(vec![]).or_insert(0) += 1;
            }
        }

        // 转换为f64 HashMap
        let mut log_ngrams: Vec<HashMap<Vec<u8>, f64>> = owned_ngrams
            .into_iter()
            .map(|map| map.into_iter().map(|(k, v)| (k, v as f64)).collect())
            .collect();
        // 从大到小遍历n-gram长度
        for i in (0..log_ngrams.len()).rev() {
            // 过滤并转换为对数
            log_ngrams[i].retain(|k, v| {
                if k.len() == i && (i <= 1 || *v >= self.min_count as f64) {
                    *v = v.ln();
                    true
                } else {
                    false
                }
            });

            // 更新i+1位置的条件概率
            if i < log_ngrams.len() - 1 {
                // 先收集需要更新的项
                let updates: Vec<(Vec<u8>, f64)> = log_ngrams[i + 1]
                    .iter()
                    .filter_map(|(k, &v)| {
                        log_ngrams[i].get(&k[..i])
                            .map(|&prefix_prob| (k.clone(), v - prefix_prob))
                    })
                    .collect();

                // 然后批量更新
                for (k, new_v) in updates {
                    if let Some(v) = log_ngrams[i + 1].get_mut(&k) {
                        *v = new_v;
                    }
                }
            }
        }
        log_ngrams
    }
}