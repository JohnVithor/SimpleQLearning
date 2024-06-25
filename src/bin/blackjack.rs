use std::env;

use clap::Parser;
use qlearning::args::Args;
use qlearning::qlearning::QLearning;
use qlearning::env::blackjack::BlackJackEnv;
use qlearning::env::{show, Env};



fn main() {
    let args = Args::parse();
    let seed = args.seed;
    let lr = args.lr;
    let gamma = args.gamma;
    let episodes = args.episodes;
    let eval_at = args.eval_at;
    let eval_for = args.eval_for;
    let show_behavior = args.print;
    
    let mut env = BlackJackEnv::new(seed);
    let mut agent: QLearning<{ BlackJackEnv::N_STATES }, { BlackJackEnv::N_ACTIONS }> =
        QLearning::new( lr, gamma, seed);
    let _ = agent.learn(&mut env, episodes, eval_at, eval_for);

    let mut wins: u32 = 0;
    let mut losses: u32 = 0;
    let mut draws: u32 = 0;
    const LOOP_LEN: usize = 1_000_000;
    for _u in 0..LOOP_LEN {
        let mut curr_action = agent.get_action(env.reset());
        loop {
            let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
            let next_action = agent.get_action(next_obs);
            curr_action = next_action;
            if terminated {
                if reward == 1.0 {
                    wins += 1;
                } else if reward == -1.0 {
                    losses += 1;
                } else {
                    draws += 1;
                }
                break;
            }
        }
    }
    println!(
        "has win-rate of {}%, loss-rate of {}% and draw-rate {}%",
        wins as f64 / LOOP_LEN as f64,
        losses as f64 / LOOP_LEN as f64,
        draws as f64 / LOOP_LEN as f64
    );
    
    if show_behavior{
        show(&mut env, agent);
    }
}
