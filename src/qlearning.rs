
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::{env::Env, policy::Policy};

pub type ValueFunction<const A: usize> = fn(&[f32; A], usize) -> f32;

#[inline(always)]
pub fn argmax<T: PartialOrd>(values: impl Iterator<Item = T>) -> usize {
    values
        .enumerate()
        .reduce(|a, b| if a.1 >= b.1 { a } else { b })
        .unwrap()
        .0
}

#[derive(Debug, Default)]
pub struct TrainResults {
    training_reward: Vec<f32>,
    training_length: Vec<usize>,
    training_error: Vec<f32>,
    mean_evaluation_reward: Vec<f32>,
    mean_evaluation_length: Vec<f32>,
}

pub struct QLearning<const S: usize, const A: usize> {
    epsilon: f32,
    rng: SmallRng,
    discount_factor: f32,
    pub policy: Policy<S, A>,
}

impl<const S: usize, const A: usize> QLearning<S, A> {
    pub fn new(
        learning_rate: f32,
        discount_factor: f32,
        seed: u64,
    ) -> Self {
        Self {
            epsilon: 1.0,
            rng: SmallRng::seed_from_u64(seed),
            discount_factor,
            policy: Policy::<S, A>::new(learning_rate),
        }
    }

    pub fn update(
        &mut self,
        curr_obs: usize,
        curr_action: usize,
        reward: f32,
        terminated: bool,
        next_obs: usize,
    ) -> f32 {
        let next_q_values = self.policy.get_values(next_obs);

        let future_q_value: f32 = next_q_values.iter().fold(f32::NAN, |acc, x| acc.max(*x));

        let curr_q_values = self.policy.get_values(curr_obs);

        let temporal_difference: f32 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        self.policy
            .update(curr_obs, curr_action, temporal_difference);

        if terminated {
            self.epsilon *= 0.9;
        }
        temporal_difference
    }

    fn should_explore(&mut self) -> bool {
        self.epsilon != 0.0 && self.rng.gen_range(0.0..1.0) <= self.epsilon
    }

    pub fn get_action(&mut self, obs: usize) -> usize {
        let values = self.policy.get_values(obs);
        if self.should_explore() {
            self.rng.gen_range(0..values.len())
        } else {
            argmax(values.iter())
        }
    }

    pub fn learn(
        &mut self,
        env: &mut dyn Env,
        n_episodes: usize,
        eval_at: usize,
        eval_for: usize,
    ) -> TrainResults {
        let mut results = TrainResults::default();
        for episode in 0..n_episodes {
            let mut action_counter: usize = 0;
            let mut epi_reward: f32 = 0.0;
            let mut curr_obs = env.reset();
            let mut curr_action = self.get_action(curr_obs);
            loop {
                action_counter += 1;

                let (next_obs, reward, done) = match env.step(curr_action) {
                    Ok(d) => d,
                    Err(e) => panic!("{e:?}"),
                };
                let next_action: usize = self.get_action(next_obs);
                let td = self.update(curr_obs, curr_action, reward, done, next_obs);
                results.training_error.push(td);
                curr_obs = next_obs;
                curr_action = next_action;
                epi_reward += reward;
                if done {
                    results.training_reward.push(epi_reward);
                    results.training_length.push(action_counter);
                    break;
                }
            }
            if episode % eval_at == 0 {
                let (r, l) = self.evaluate(env, eval_for);
                let mr: f32 = r.iter().sum::<f32>() / r.len() as f32;
                let ml: f32 = l.iter().sum::<usize>() as f32 / l.len() as f32;
                results.mean_evaluation_reward.push(mr);
                results.mean_evaluation_length.push(ml);
            }
        }

        results
    }

    pub fn evaluate(&mut self, env: &mut dyn Env, episodes: usize) -> (Vec<f32>, Vec<usize>) {
        let mut rewards = Vec::new();
        let mut lengths = Vec::new();
        for _ in 0..episodes {
            let mut obs = env.reset();
            let mut reward: f32 = 0.0;
            let mut steps: usize = 0;
            loop {
                let action = self.get_action(obs);
                let Ok((next_obs, r, done)) = env.step(action) else {
                    panic!("Error in step")
                };
                obs = next_obs;
                reward += r;
                steps += 1;
                if done {
                    rewards.push(reward);
                    lengths.push(steps);
                    break;
                }
            }
        }
        (rewards, lengths)
    }

    pub fn reset(&mut self) {
        self.policy.reset();
        self.epsilon = 1.0;
    }
}
