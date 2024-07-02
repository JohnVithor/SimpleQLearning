#[derive(Debug, Clone)]
pub struct Policy<const S: usize, const A: usize> {
    learning_rate: f32,
    policy: [[f32; A]; S],
}

impl<const S: usize, const A: usize> Policy<S, A> {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            policy: [[0.0; A]; S],
        }
    }

    pub fn get_values(&mut self, obs: usize) -> [f32; A] {
        self.policy[obs]
    }

    pub fn update(&mut self, obs: usize, action: usize, td_target: f32) -> f32 {
        let td_error = td_target - self.policy[obs][action];
        self.policy[obs][action] += self.learning_rate * td_error;
        td_error
    }

    pub fn reset(&mut self) {
        self.policy.fill([0.0; A]);
    }
}
