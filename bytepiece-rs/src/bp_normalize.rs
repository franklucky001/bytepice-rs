use unicode_normalization::UnicodeNormalization;
use regex::Regex;
pub struct TextNormalizer {
    max_len: usize,
    isolate_digits: bool,
}

impl TextNormalizer {
    const MAX_REGEX_LEN: usize = 1000;
    pub fn new(max_len: usize, isolate_digits: bool) -> Self {
        Self {
            max_len,
            isolate_digits,
        }
    }

    fn prepare(&self, text: &str) -> (String, Regex) {
        // Unicode NFC 标准化
        let normalized = text.nfc().collect::<String>();

        // 构建正则表达式
        let regex_pattern = if self.max_len > 0 {
            if self.isolate_digits {
                format!(
                    r"\d|[^\n\d]{{0,{}}}\n{{1,100}}|[^\n\d]{{1,{}}}",
                    self.max_len,
                    self.max_len
                )
            } else {
                format!(
                    r".{{0,{}}}\n{{1,100}}|.{{1,{}}}",
                    self.max_len,
                    self.max_len
                )
            }
        } else {
            if self.isolate_digits {
                r"\d|[^\n\d]*\n+|[^\n\d]+".to_string()
            } else {
                r".*\n+|.+".to_string()
            }
        };// 编译正则表达式
        let regex_digits = Regex::new(&regex_pattern).expect("regex failed to compile");

        (normalized, regex_digits)
    }

    pub fn normalize(&self, text: &str) -> Vec<Vec<u8>> {
        if self.max_len < Self::MAX_REGEX_LEN {
            let (normalized, regex_digits) = self.prepare(text);
            // 查找所有匹配并转换为字节向量
            regex_digits.find_iter(&normalized)
                .map(move |m| m.as_str().as_bytes().to_vec())
                .collect()
        }else {
            self.normalize_custom_split(text)
        }
    }

    fn normalize_custom_split(&self, text: &str) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut newlines = String::new();

        for c in text.chars() {
            if c == '\n' {
                if !current.is_empty() {
                    self.add_chunks(&mut result, &current);
                    current.clear();
                }
                newlines.push(c);
            } else {
                if !newlines.is_empty() {
                    result.push(newlines.as_bytes().to_vec());
                    newlines.clear();
                }

                if self.isolate_digits && c.is_ascii_digit() {
                    if !current.is_empty() {
                        self.add_chunks(&mut result, &current);
                        current.clear();
                    }
                    result.push(vec![c as u8]);
                } else {
                    current.push(c);
                    if current.len() >= self.max_len {
                        self.add_chunks(&mut result, &current);
                        current.clear();
                    }
                }
            }
        }

        if !current.is_empty() {
            self.add_chunks(&mut result, &current);
        }
        if !newlines.is_empty() {
            result.push(newlines.as_bytes().to_vec());
        }

        result
    }
    fn add_chunks(&self, result: &mut Vec<Vec<u8>>, text: &str) {
        if text.len() <= self.max_len {
            result.push(text.as_bytes().to_vec());
            return;
        }

        let mut current = Vec::new();
        let mut byte_count = 0;

        for c in text.chars() {
            let char_bytes = c.len_utf8();
            if byte_count + char_bytes > self.max_len {
                if !current.is_empty() {
                    result.push(current);
                    current = Vec::new();
                    byte_count = 0;
                }
            }
            let mut buf = [0; 4];
            c.encode_utf8(&mut buf);
            current.extend_from_slice(&buf[..char_bytes]);
            byte_count += char_bytes;
        }

        if !current.is_empty() {
            result.push(current);
        }
    }
}

// 便捷函数
pub(crate) fn _normalize(text: &str, max_len: usize, isolate_digits: bool) -> Vec<Vec<u8>> {
    let normalizer = TextNormalizer::new(max_len, isolate_digits);
    normalizer.normalize(text)
}

// 定义宏
#[macro_export]
macro_rules! normalize {
    ($text:expr) => {
        $crate::bp_normalize::_normalize($text, 0, false)
    };
    ($text:expr, $max_len:expr) => {
        $crate::bp_normalize::_normalize($text, $max_len, false)
    };
    ($text:expr, $max_len:expr, $isolate_digits:expr) => {
        $crate::bp_normalize::_normalize($text, $max_len, $isolate_digits)
    };
}

