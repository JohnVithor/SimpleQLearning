use crate::qlearning::QLearning;

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

pub fn show<const T: usize, const A: usize>(
    env: &mut dyn Env,
    mut agent: QLearning<T, A>,
) {
    let mut done = false;
    while !done {
        let mut obs = env.reset();
        loop {
            println!("{}", env.render());
            let action = agent.get_action(obs);
            let Ok((next_obs, _, d)) = env.step(action) else {
                panic!("Error in step")
            };
            obs = next_obs;
            if d {
                done = true;
                println!("{}", env.render());
                break;
            }
        }
    }
}