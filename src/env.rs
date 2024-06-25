pub mod blackjack;
pub mod clifwalking;

#[derive(Debug, Copy, Clone)]
pub enum EnvError {
    EnvNotReady,
    InvalidAction,
}

pub trait Env {
    fn reset(&mut self) -> usize;
    fn step(&mut self, action: usize) -> Result<(usize, f32, bool), EnvError>;
    fn render(&self) -> String;
}
