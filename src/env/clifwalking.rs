use crate::env::{Env, EnvError};

#[derive(Debug, Clone)]
pub struct CliffWalkingEnv {
    ready: bool,
    obs: [[(usize, f32, bool); 4]; 48],
    player_pos: usize,
    max_steps: usize,
    curr_step: usize,
}

impl CliffWalkingEnv {
    pub const N_ACTIONS: usize = 4;
    pub const N_STATES: usize = 48;
    const NCOL: usize = 12;
    const NROW: usize = 4;
    const START_POSITION: usize = 36;
    const CLIFF_POSITIONS: [usize; 10] = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46];
    const GOAL_POSITION: usize = 47;
    const MAP: &'static str = "____________\n____________\n____________\n_!!!!!!!!!!_";

    fn inc(row: usize, col: usize, a: usize) -> (usize, usize) {
        let new_col: usize;
        let new_row: usize;
        if a == 0 {
            // left
            new_col = if col != 0 { (col - 1).max(0) } else { 0 };
            new_row = row;
        } else if a == 1 {
            // down
            new_col = col;
            new_row = (row + 1).min(Self::NROW - 1);
        } else if a == 2 {
            // right
            new_col = (col + 1).min(Self::NCOL - 1);
            new_row = row;
        } else if a == 3 {
            // up
            new_col = col;
            new_row = if row != 0 { (row - 1).max(0) } else { 0 };
        } else {
            return (row, col);
        }
        (new_row, new_col)
    }

    fn update_probability_matrix(row: usize, col: usize, action: usize) -> (usize, f32, bool) {
        let (newrow, newcol) = Self::inc(row, col, action);
        let newstate: usize = newrow * Self::NCOL + newcol;
        let win: bool = newstate == Self::GOAL_POSITION;
        let lose: bool = Self::CLIFF_POSITIONS.contains(&newstate);
        let reward: f32 = if lose { -100.0 } else { -1.0 };
        (newstate, reward, lose || win)
    }

    pub fn new(max_steps: usize) -> Self {
        let mut initial_state_distrib = [0.0; Self::NROW * Self::NCOL];
        initial_state_distrib[Self::START_POSITION] = 1.0;
        let mut obs = [[(0, 0.0, false); Self::N_ACTIONS]; Self::NROW * Self::NCOL];
        for row in 0..Self::NROW {
            for col in 0..Self::NCOL {
                for a in 0..Self::N_ACTIONS {
                    obs[row * Self::NCOL + col][a] = Self::update_probability_matrix(row, col, a);
                }
            }
        }

        Self {
            ready: false,
            obs,
            player_pos: 0,
            max_steps,
            curr_step: 0,
        }
    }
}

impl Env for CliffWalkingEnv {
    fn reset(&mut self) -> usize {
        self.player_pos = Self::START_POSITION;
        self.ready = true;
        self.curr_step = 0;
        self.player_pos
    }

    fn step(&mut self, action: usize) -> Result<(usize, f32, bool), EnvError> {
        if !self.ready {
            return Err(EnvError::EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((0, -1.0, true));
        }
        self.curr_step += 1;
        if action > 3 {
            return Err(EnvError::InvalidAction);
        }
        let obs: (usize, f32, bool) = self.obs[self.player_pos][action];
        self.player_pos = obs.0;
        if obs.2 {
            self.ready = false;
        }
        Ok(obs)
    }

    fn render(&self) -> String {
        let mut new_map: String = <&str>::clone(&Self::MAP).to_string();
        let mut pos: usize = self.player_pos;
        for (i, _) in new_map.match_indices('\n') {
            if pos >= i {
                pos += 1;
            }
        }
        new_map.replace_range(pos..pos + 1, "@");
        new_map
    }
}
