use qlearning::clifwalking::CliffWalkingEnv;
use qlearning::qlearning::{qlearning, sarsa, QLearning};
use qlearning::Env;

fn main() {
    let seed = 10;
    let lr = 0.1;
    let gamma = 0.99999;
    let steps = 100_000;
    let eval_at = 1000;
    let eval_for = 10;

    let mut env = CliffWalkingEnv::new(20);
    let mut agent: QLearning<{ CliffWalkingEnv::N_STATES }, { CliffWalkingEnv::N_ACTIONS }> =
        QLearning::new(sarsa, lr, gamma, seed);
    let results = agent.learn(&mut env, steps, eval_at, eval_for);
    println!("sarsa {:?}", results.mean_evaluation_reward);

    show_execution(&mut env, agent);

    let mut agent: QLearning<{ CliffWalkingEnv::N_STATES }, { CliffWalkingEnv::N_ACTIONS }> =
        QLearning::new(qlearning, lr, gamma, seed);
    let results = agent.learn(&mut env, steps, eval_at, eval_for);
    println!("qlearning {:?}", results.mean_evaluation_reward);

    show_execution(&mut env, agent);
}

fn show_execution<const T: usize, const A: usize>(
    env: &mut CliffWalkingEnv,
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
