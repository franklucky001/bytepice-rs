use pyo3::prelude::*;
use pyo3::types::PyBytes;
use bytepiece_rs::Tokenizer;


// Python 类包装器
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    // 构造函数
    #[pyo3(signature = (filepath, seed=None))]
    #[staticmethod]
    fn from_json(filepath: &str, seed: Option<u64>) -> Self {
        let inner = Tokenizer::from_file(filepath, seed);
        PyTokenizer { inner }
    }

    // tokenize 方法
    #[pyo3(signature = (text, alpha=-1.0))]
    fn tokenize<'py>(&self, py: Python<'py>, text: &str, alpha: f64) -> Bound<'py, PyBytes> {
        let ret = self.inner
            .tokenize(text, alpha);
        PyBytes::new(py, ret.as_slice())
    }
    // batch_tokenize 方法
    #[pyo3(signature = (texts, alpha=-1.0))]
    fn batch_tokenize<'py >(&self, py: Python<'py>, texts:Vec<String>, alpha: f64) -> Vec<Bound<'py, PyBytes>> {
        self.inner
            .batch_tokenize(texts, alpha)
            .into_iter()
            .map(|it|PyBytes::new(py, it.as_slice()))
            .collect()
    }
}

// Python 模块定义
#[pymodule]
fn bytepiece_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}

