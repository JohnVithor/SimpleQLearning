use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::clifwalking::CliffWalkingEnv;

pub type ValueFunction<const A: usize> = fn(&[f32; A], usize) -> f32;

#[inline(always)]
pub fn argmax<T: PartialOrd>(values: impl Iterator<Item = T>) -> usize {
    values
        .enumerate()
        .reduce(|a, b| if a.1 >= b.1 { a } else { b })
        .unwrap()
        .0
}

pub fn sarsa<const A: usize>(next_q_values: &[f32; A], next_action: usize) -> f32 {
    next_q_values[next_action]
}

pub fn qlearning<const A: usize>(next_q_values: &[f32; A], _next_action: usize) -> f32 {
    next_q_values.iter().fold(f32::NAN, |acc, x| acc.max(*x))
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
    value_function: ValueFunction<A>,
    learning_rate: f32,
    discount_factor: f32,
    policy: [[f32; A]; S],
}

impl<const S: usize, const A: usize> QLearning<S, A> {
    const DEFAULT_VALUES: [f32; A] = [0.0; A];
    pub fn new(
        value_function: ValueFunction<A>,
        learning_rate: f32,
        discount_factor: f32,
        seed: u64,
    ) -> Self {
        Self {
            epsilon: 1.0,
            rng: SmallRng::seed_from_u64(seed),
            value_function,
            learning_rate,
            discount_factor,
            policy: [[0.0; A]; S],
        }
    }

    pub fn update(
        &mut self,
        curr_obs: usize,
        curr_action: usize,
        reward: f32,
        terminated: bool,
        next_obs: usize,
        next_action: usize,
    ) -> f32 {
        let next_q_values = self.policy.get(next_obs).unwrap_or(&Self::DEFAULT_VALUES);

        let future_q_value: f32 = (self.value_function)(next_q_values, next_action);

        let curr_q_values = self.policy.get(curr_obs).unwrap_or(&Self::DEFAULT_VALUES);
        let temporal_difference: f32 = reward
            + if terminated {
                0.0
            } else {
                self.discount_factor * future_q_value
            }
            - curr_q_values[curr_action];

        let value = self.policy.get(curr_obs).unwrap_or(&Self::DEFAULT_VALUES)[curr_action];
        self.policy.get_mut(curr_obs).unwrap()[curr_action] =
            value + self.learning_rate * temporal_difference;

        if terminated {
            self.epsilon *= 0.9;
        }
        temporal_difference
    }

    fn should_explore(&mut self) -> bool {
        self.epsilon != 0.0 && self.rng.gen_range(0.0..1.0) <= self.epsilon
    }

    pub fn get_action(&mut self, obs: usize) -> usize {
        if self.should_explore() {
            let values = self.policy.get(obs).unwrap_or(&Self::DEFAULT_VALUES);
            self.rng.gen_range(0..values.len())
        } else {
            let values = self.policy.get(obs).unwrap_or(&Self::DEFAULT_VALUES);
            argmax(values.iter())
        }
    }

    pub fn learn(
        &mut self,
        env: &mut CliffWalkingEnv,
        steps: usize,
        eval_at: usize,
        eval_for: usize,
    ) -> TrainResults {
        let mut results = TrainResults::default();
        let mut curr_obs = env.reset();
        let mut curr_action = self.get_action(curr_obs);
        let mut epi_reward: f32 = 0.0;
        let mut episode = 0;
        for step in 1..=steps {
            let r = env.step(curr_action);

            let (next_obs, reward, done) = match r {
                Ok(d) => d,
                Err(e) => panic!("{e:?}"),
            };
            let next_action = self.get_action(next_obs);
            let td = self.update(curr_obs, curr_action, reward, done, next_obs, next_action);
            results.training_error.push(td);
            curr_obs = next_obs;
            curr_action = next_action;
            epi_reward += reward;
            if done {
                if episode % eval_at == 0 {
                    let (r, l) = self.evaluate(env, eval_for);
                    let mr: f32 = r.iter().sum::<f32>() / r.len() as f32;
                    let ml: f32 = l.iter().sum::<usize>() as f32 / l.len() as f32;
                    results.mean_evaluation_reward.push(mr);
                    results.mean_evaluation_length.push(ml);
                }
                results.training_reward.push(epi_reward);
                curr_obs = env.reset();
                curr_action = self.get_action(curr_obs);
                results.training_length.push(step);
                episode += 1;
            }
        }
        results
    }

    pub fn evaluate(
        &mut self,
        env: &mut CliffWalkingEnv,
        episodes: usize,
    ) -> (Vec<f32>, Vec<usize>) {
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
        self.policy = [[0.0; A]; S];
        self.epsilon = 1.0;
    }
}
