use std::collections::HashSet;
use std::hash::Hash;
use aho_corasick::{AhoCorasick, FindOverlappingIter, Input};

pub(crate) struct Automaton<P, V> {
    patterns: Vec<P>,
    values: Vec<V>,
    ac: AhoCorasick,
}

impl<P, V> Automaton<P, V>
where P: AsRef<[u8]>
{
    pub fn new(patterns: Vec<P>, values: Vec<V>) -> Self {
        let ac = AhoCorasick::new(&patterns).expect("AhoCorasick::new failed");
        Self { patterns, values, ac}
    }

    pub fn iter<'a, 'h, I: Into<Input<'h>>>(&'a self, input: I) -> AutomatonIter<'a, 'h, P, V>{
        AutomatonIter::new(&self.patterns, &self.values, self.ac.find_overlapping_iter(input))
    }

    pub fn get(&self, pattern: P) -> Option<&V> {
        // todo ac.find mat???
        let pos = self.patterns.iter().position(|p| p.as_ref() == pattern.as_ref()).expect("unexpected pattern");
        self.values.get(pos)
    }
}


pub(crate) struct AutomatonIter<'a, 'h, P, V> {
    patterns: &'a Vec<P>,
    values: &'a Vec<V>,
    iter: FindOverlappingIter<'a, 'h>,
}

impl<'a, 'h, P, V> AutomatonIter<'a, 'h, P, V> {
    pub fn new(patterns: &'a Vec<P>, values: &'a Vec<V>, iter: FindOverlappingIter<'a, 'h>) -> Self {
        Self { patterns, values, iter }
    }
}

impl<'a, 'h, P, V> Iterator for AutomatonIter<'a, 'h, P, V> {
    type Item = (usize, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            None => None,
            Some(mat) => {
                let pid = mat.pattern().as_usize();
                Some((mat.end(), &self.values[pid]))
            }
        }
    }
}

pub(crate) struct AutomatonBuilder<P, V> {
    patterns: Vec<P>,
    values: Vec<V>,
    state: HashSet<P>,
}

impl<P: AsRef<[u8]>, V> AutomatonBuilder<P, V> {
    pub fn new() -> Self {
        Self { patterns: Vec::new(), values: Vec::new(), state: HashSet::new() }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self { patterns: Vec::with_capacity(capacity), values: Vec::with_capacity(capacity), state: HashSet::with_capacity(capacity)}
    }
    pub fn insert(&mut self, pattern: P, value: V)
    where P: AsRef<[u8]> + Eq + Hash + Clone
    {
        if !self.state.contains(&pattern) {
            self.state.insert(pattern.clone());
            self.patterns.push(pattern);
            self.values.push(value);
        }
    }
    pub fn builder_automaton(self)-> Automaton<P, V>{
        Automaton::new(self.patterns, self.values)
    }
}