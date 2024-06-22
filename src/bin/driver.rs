use qlearning::clifwalking::CliffWalkingEnv;
use qlearning::qlearning::{qlearning, QLearning};

fn main() {
    let mut env = CliffWalkingEnv::new(1000);
    let mut agent: QLearning<{ CliffWalkingEnv::N_STATES }, { CliffWalkingEnv::N_ACTIONS }> =
        QLearning::new(qlearning, 0.1, 0.9, 0);
    let results = agent.learn(&mut env, 1000, 100, 10);
    println!("{:?}", results);
}
