use clap::Parser;

use qlearning::args::Args;
use qlearning::qlearning::QLearning;
use qlearning::env::clifwalking::CliffWalkingEnv;
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

    let mut env = CliffWalkingEnv::new(100);
    let mut agent: QLearning<{ CliffWalkingEnv::N_STATES }, { CliffWalkingEnv::N_ACTIONS }> =
        QLearning::new(lr, gamma, seed);
    let _results = agent.learn(&mut env, episodes, eval_at, eval_for);
    // println!("sarsa {:?}", results.mean_evaluation_reward);
    let results = agent.evaluate(&mut env, eval_for);
    println!(
        "qlearning {:?}",
        results.0.iter().sum::<f32>() / eval_for as f32
    );

    if show_behavior{
        show(&mut env, agent);
    }

}

