use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use rand::rngs::StdRng;
use rand::prelude::*;

static GLOBAL_CHOOSER: OnceLock<Mutex<RandomChooser>> = OnceLock::new();
pub(crate) struct  RandomChooser{
    rng: StdRng,
}
impl RandomChooser {
    pub fn init_global_chooser(seed: Option<u64>) {
        GLOBAL_CHOOSER.get_or_init(|| {
            Mutex::new(RandomChooser::new(init_random_seed(seed)))
        });
    }
    fn get_chooser() -> &'static Mutex<RandomChooser>{
        GLOBAL_CHOOSER.get().expect("Random Chooser is not initialized!")
    }
    pub fn new(rng: StdRng) -> Self{
        Self { rng}
    }
    // choice实现
    #[inline]
    fn choice(&mut self, x: f64, y: f64) -> bool {
        let rng = &mut self.rng;
        rng.gen::<f64>() < (x - y).exp()
    }

    #[inline]
    fn batch_choice(&mut self, xs: &[f64], ys: &[f64]) -> Vec<bool> {
        let rng = &mut self.rng;
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| (rng.gen::<f64>() < (x - y).exp()))
            .collect()
    }
}

// 默认使用系统时间作为种子初始化
fn init_random_seed(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => {
            StdRng::seed_from_u64(seed)
        },
        None => {
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            StdRng::seed_from_u64(seed)
        }
    }
}

pub(crate) fn choice(x: f64, y: f64) -> bool {
    let chooser = RandomChooser::get_chooser();
    chooser.lock().unwrap().choice(x, y)
}