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

    pub fn predict(&mut self, obs: usize) -> [f32; A] {
        self.policy[obs]
    }

    pub fn get_values(&mut self, obs: usize) -> [f32; A] {
        self.policy[obs]
    }

    pub fn update(&mut self, obs: usize, action: usize, temporal_difference: f32) -> f32 {
        let td = self.learning_rate * temporal_difference;
        self.policy[obs][action] += td;
        td
    }

    pub fn reset(&mut self) {
        self.policy.fill([0.0; A]);
    }
}
