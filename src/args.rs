use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// seed
    #[arg(short, long, default_value_t = 0)]
    pub seed: u64,

    /// learning rate
    #[arg(short, long, default_value_t = 0.1)]
    pub lr: f32,

    /// gamma (discount factor)
    #[arg(short, long, default_value_t = 0.95)]
    pub gamma: f32,

    /// number of training episodes
    #[arg(short, long, default_value_t = 1_000)]
    pub episodes: usize,

    /// eval agent at every X episodes
    #[arg(short, long, default_value_t = 100)]
    pub eval_at: usize,

    /// eval agent for X episodes
    #[arg(short, long, default_value_t = 10)]
    pub eval_for: usize,

    /// show agent behavior after training
    #[arg(short, long)]
    pub print: bool,
}
