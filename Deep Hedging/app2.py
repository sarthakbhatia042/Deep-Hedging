import torch
import torch.nn as nn
import torch.nn.functional as F

class CashInvariantDBH(nn.Module):
    def __init__(self, state_dim, action_dim, risk_aversion=1.0):
        super().__init__()
        self.lam = risk_aversion
        
        # ACTOR: Decides the hedge ratio based on the state
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128), # LayerNorm stabilizes RL training
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid() # Hedge ratio between 0 and 1
        )
        
        # Q-CRITIC: Evaluates the specific Action taken in the State
        # Input = State + Action concatenated
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_action(self, state):
        return self.actor(state)

    def compute_loss(self, state, action, cash_flow, next_state, done):
        """
        cash_flow: Immediate costs incurred (-fees). 
                   If done=True, this must include the terminal option payoff.
        """
        # 1. Current Q-Value Estimate
        q_curr = self.critic(torch.cat([state, action], dim=-1))
        
        # 2. Target Q-Value using the Actor's next planned move
        with torch.no_grad():
            next_action = self.actor(next_state)
            q_next = self.critic(torch.cat([next_state, next_action], dim=-1))
            
            # Cash-Invariant Bellman Target
            target_q = cash_flow + (1 - done) * q_next

        # 3. CRITIC LOSS: Numerically Stable Entropic Risk Loss
        # We want to minimize the Entropic Risk of the TD-Error: (target_q - q_curr)
        td_error = target_q - q_curr
        scaled_error = -self.lam * td_error
        
        # LogSumExp trick to prevent float overflow during exp()
        max_val = torch.max(scaled_error)
        critic_loss = max_val + torch.log(torch.mean(torch.exp(scaled_error - max_val))) / self.lam

        # 4. ACTOR LOSS: Maximize the Q-Value (Minimize Risk/Liability)
        # The actor is trained to output actions that the critic scores highly
        current_actor_action = self.actor(state)
        actor_loss = -self.critic(torch.cat([state, current_actor_action], dim=-1)).mean()
        
        return actor_loss, critic_loss