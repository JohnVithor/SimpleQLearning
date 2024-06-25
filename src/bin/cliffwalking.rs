use qlearning::agent::{qlearning, sarsa, Agent};
use qlearning::env::clifwalking::CliffWalkingEnv;
use qlearning::env::Env;

fn main() {
    let seed = 1;
    let lr = 0.1;
    let gamma = 0.9;
    let episodes = 1_000;
    let eval_at = 10;
    let eval_for = 10;

    let mut env = CliffWalkingEnv::new(100);
    let mut agent: Agent<{ CliffWalkingEnv::N_STATES }, { CliffWalkingEnv::N_ACTIONS }> =
        Agent::new(sarsa, lr, gamma, seed);
    let _results = agent.learn(&mut env, episodes, eval_at, eval_for);
    // println!("sarsa {:?}", results.mean_evaluation_reward);
    let results = agent.evaluate(&mut env, eval_for);
    println!(
        "sarsa {:?}",
        results.0.iter().sum::<f32>() / eval_for as f32
    );
    // show_execution(&mut env, agent);

    let mut agent: Agent<{ CliffWalkingEnv::N_STATES }, { CliffWalkingEnv::N_ACTIONS }> =
        Agent::new(qlearning, lr, gamma, seed);
    let _results = agent.learn(&mut env, episodes, eval_at, eval_for);
    let results = agent.evaluate(&mut env, eval_for);
    println!(
        "qlearning {:?}",
        results.0.iter().sum::<f32>() / eval_for as f32
    );
    // println!("qlearning {:?}", results.mean_evaluation_reward);

    // show_execution(&mut env, agent);
}

fn show_execution<const T: usize, const A: usize>(
    env: &mut CliffWalkingEnv,
    mut agent: Agent<T, A>,
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
