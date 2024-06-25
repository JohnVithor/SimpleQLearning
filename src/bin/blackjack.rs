use qlearning::agent::{sarsa, Agent};
use qlearning::env::blackjack::BlackJackEnv;
use qlearning::env::Env;

fn main() {
    let mut env = BlackJackEnv::new(40);
    let mut agent: Agent<{ BlackJackEnv::N_STATES }, { BlackJackEnv::N_ACTIONS }> =
        Agent::new(sarsa, 0.1, 0.9, 0);
    let _ = agent.learn(&mut env, 10_000, 1000, 10);

    let mut wins: u32 = 0;
    let mut losses: u32 = 0;
    let mut draws: u32 = 0;
    const LOOP_LEN: usize = 1000000;
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
}
