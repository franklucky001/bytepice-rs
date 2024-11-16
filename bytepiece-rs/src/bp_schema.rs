use std::collections::HashMap;
use std::hash::Hash;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug)]
pub struct PieceState{
    pub(crate) value: usize,
    piece: Vec<u8>,
    pub(crate) pid: usize,
}
// 为了处理元组格式的序列化/反序列化
impl<'de> Deserialize<'de> for PieceState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>
    {
        use serde::de::Error;

        // 首先反序列化为JSON Value
        let value = Value::deserialize(deserializer)?;

        if let Value::Array(arr) = value {
            if arr.len() != 3 {
                return Err(D::Error::custom("expected array of length 3"));
            }

            // 解析value
            let value = arr[0].as_u64()
                .ok_or_else(|| D::Error::custom("invalid value"))?;

            // 解析char（处理特殊情况）
            let piece = match &arr[1] {
                Value::String(s) => {
                    if s.is_empty() {
                        Vec::new()
                    } else {
                        s.as_bytes().to_vec()
                    }
                },
                _ => return Err(D::Error::custom("expected string for char")),
            };

            // 解析pid
            let pid = arr[2].as_u64()
                .ok_or_else(|| D::Error::custom("invalid pid"))?;

            Ok(PieceState {
                value: value as usize,
                piece,
                pid: pid as usize,
            })
        } else {
            Err(D::Error::custom("expected array"))
        }
    }
}

impl Serialize for PieceState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;

        // 创建一个长度为3的序列
        let mut seq = serializer.serialize_seq(Some(3))?;
        // 序列化value
        seq.serialize_element(&self.value)?;
        // 序列化char（特殊处理\0）
        let piece_str = if self.piece.is_empty() {
            "\u{0000}".to_string()
        } else {
            String::from_utf8_lossy(&*self.piece).into_owned()
        };
        seq.serialize_element(&piece_str)?;
        // 序列化pid
        seq.serialize_element(&self.pid)?;
        seq.end()
    }
}

impl PieceState {
    pub fn new(value: usize, piece: Vec<u8>, pid: usize) -> PieceState {
        PieceState{value, piece, pid}
    }
}

pub trait PieceDecoder{
    fn decode(self) -> Vec<u8>;
    fn encode(&self) -> String;
}

impl PieceDecoder for String {
    fn decode(self) -> Vec<u8> {
        BASE64_STANDARD.decode(self.as_bytes())
            .expect("piece base64 decode failed")
    }
    fn encode(&self) -> String {
        BASE64_STANDARD.encode(self.as_str())
    }
}

impl PieceDecoder for Vec<u8> {
    fn decode(self) -> Vec<u8> {
        self
    }
    fn encode(&self) -> String {
        BASE64_STANDARD.encode(self.as_slice())
    }
}

impl PieceDecoder for &Vec<u8> {
    fn decode(self) -> Vec<u8> {
        self.clone()
    }
    fn encode(&self) -> String {
        BASE64_STANDARD.encode(self.as_slice())
    }
}

impl PieceDecoder for &[u8] {
    fn decode(self) -> Vec<u8> {
        self.to_vec()
    }
    fn encode(&self) -> String {
        BASE64_STANDARD.encode(self)
    }
}

pub(crate) struct PieceMap<P>(HashMap<P, (usize, Vec<u8>, usize)>);

impl<P> PieceMap<P>
where P: PieceDecoder + Clone + Eq + Hash
{
    pub(crate) fn new(mapper: HashMap<P, (usize, Vec<u8>, usize)>) -> Self {
        Self(mapper)
    }
    pub(crate) fn inner(self) -> HashMap<P, PieceState> {
        self.0.into_iter().map(|(k, v)|(k, PieceState::new(v.0, v.1, v.2))).collect()
    }
}

impl<'a> From<&'a HashMap<Vec<u8>, usize>> for PieceMap<&'a Vec<u8>> {
    fn from(value: &'a HashMap<Vec<u8>, usize>) -> Self {
        let pieces = value.into_iter().collect::<Vec<_>>();
        let mapper = pieces
            .into_iter()
            .enumerate()
            .map(|(i, (k, v))|(k, (i+3, k.to_owned(), *v)))
            .collect();
        Self(mapper)
    }
}