import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import matplotlib.ticker as ticker

# ==========================================
# 0. PUBLICATION STYLE SETUP
# ==========================================
def setup_plot_style():
    """Configures Matplotlib for top-tier conference quality."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.figsize': (8, 6),
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': ':',
        'grid.color': 'black',
        'lines.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'font.family': 'sans-serif', # Or 'serif' for Times New Roman style
    })

# Define the exact Green from your LaTeX (\definecolor{rljgreen}{RGB}{0, 100, 0})
RLJ_GREEN = '#006400' 
GRAY_BASELINE = '#808080'
SYCOPHANT_RED = '#D32F2F'
CONTEXTUAL_ORANGE = '#F57C00'
HONEST_GREEN = '#388E3C'

# ==========================================
# 1. THE AGENTS (Unchanged Logic)
# ==========================================

class ContextualTrustModel:
    def __init__(self, context_dim):
        self.model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, 
                                   fit_intercept=True, learning_rate='optimal', random_state=42)
        self.is_initialized = False
        self.context_dim = context_dim

    def predict_trust(self, context):
        if not self.is_initialized:
            return 0.5
        X = context.reshape(1, -1)
        return self.model.predict_proba(X)[0, 1]

    def update(self, context, is_truthful):
        X = context.reshape(1, -1)
        y = np.array([is_truthful])
        self.model.partial_fit(X, y, classes=np.array([0, 1]))
        self.is_initialized = True

class ESALinUCB:
    def __init__(self, num_arms, context_dim, num_evaluators, alpha=0.5, lambda_reg=1.0, axiom_prob=0.1):
        self.num_arms = num_arms
        self.d = context_dim
        self.M = num_evaluators
        self.alpha = alpha
        self.axiom_prob = axiom_prob
        self.A_inv = [np.eye(self.d) / lambda_reg for _ in range(num_arms)] 
        self.b = [np.zeros(self.d) for _ in range(num_arms)]
        self.trust_models = [ContextualTrustModel(self.d) for _ in range(self.M)]
        self.log_trust_weights = []

    def select_action(self, context):
        ucb_scores = []
        for a in range(self.num_arms):
            theta_hat = self.A_inv[a] @ self.b[a]
            confidence = self.alpha * np.sqrt(context.T @ self.A_inv[a] @ context)
            est_reward = context.T @ theta_hat
            ucb_scores.append(est_reward + confidence)
        return np.argmax(ucb_scores)

    def run_step(self, context, action, social_feedback_vector, ground_truth_oracle=None):
        raw_trust = np.array([tm.predict_trust(context) for tm in self.trust_models])
        sum_trust = np.sum(raw_trust)
        weights = raw_trust / sum_trust if sum_trust > 0 else np.ones(self.M)/self.M
        self.log_trust_weights.append(weights)

        axiom_triggered = np.random.rand() < self.axiom_prob
        if axiom_triggered and ground_truth_oracle is not None:
            z_t = ground_truth_oracle(context, action)
            epsilon = 0.1
            for m in range(self.M):
                y_m = social_feedback_vector[m]
                is_truthful = 1 if abs(y_m - z_t) < epsilon else 0
                self.trust_models[m].update(context, is_truthful)
        
        trusted_reward = np.dot(weights, social_feedback_vector)
        update_weight = np.max(raw_trust)
        x = context.reshape(-1, 1)
        A_inv = self.A_inv[action]
        numerator = update_weight * (A_inv @ x @ x.T @ A_inv)
        denominator = 1.0 + update_weight * (x.T @ A_inv @ x)
        self.A_inv[action] = A_inv - (numerator / denominator)
        self.b[action] += update_weight * trusted_reward * context

class StandardLinUCB:
    def __init__(self, num_arms, context_dim, alpha=0.5, lambda_reg=1.0):
        self.num_arms = num_arms
        self.d = context_dim
        self.alpha = alpha
        self.A_inv = [np.eye(self.d) / lambda_reg for _ in range(num_arms)]
        self.b = [np.zeros(self.d) for _ in range(num_arms)]

    def select_action(self, context):
        ucb_scores = []
        for a in range(self.num_arms):
            theta_hat = self.A_inv[a] @ self.b[a]
            confidence = self.alpha * np.sqrt(context.T @ self.A_inv[a] @ context)
            est_reward = context.T @ theta_hat
            ucb_scores.append(est_reward + confidence)
        return np.argmax(ucb_scores)

    def run_step(self, context, action, social_feedback_vector):
        avg_feedback = np.mean(social_feedback_vector)
        x = context.reshape(-1, 1)
        A_inv = self.A_inv[action]
        numerator = (A_inv @ x @ x.T @ A_inv)
        denominator = 1.0 + (x.T @ A_inv @ x)
        self.A_inv[action] = A_inv - (numerator / denominator)
        self.b[action] += avg_feedback * context

# ==========================================
# 2. ENVIRONMENT (Unchanged Logic)
# ==========================================

class ScalableJekyllHydeEnvironment:
    def __init__(self, context_dim, num_arms, num_evaluators=10):
        self.d = context_dim
        self.k = num_arms
        self.M = num_evaluators
        self.theta_star = [np.random.randn(context_dim) for _ in range(num_arms)]
        self.theta_star = [t / np.linalg.norm(t) for t in self.theta_star]
        self.types = []
        n_honest = max(1, int(0.2 * self.M))
        n_context = int(0.3 * self.M)
        n_sycophant = self.M - n_honest - n_context
        self.types = ([0]*n_honest) + ([1]*n_context) + ([2]*n_sycophant)
        self.types = sorted(self.types) 
        self.lie_vectors = {}
        for i, t in enumerate(self.types):
            if t == 1: 
                v = np.random.randn(context_dim)
                self.lie_vectors[i] = v / np.linalg.norm(v)

    def get_context(self):
        return np.random.rand(self.d)

    def get_true_reward(self, context, action):
        r = np.dot(self.theta_star[action], context)
        return np.clip(r + np.random.normal(0, 0.01), 0, 1)

    def get_social_feedback(self, context, action, true_reward):
        feedback = []
        for i in range(self.M):
            etype = self.types[i]
            if etype == 0: # Honest
                fb = true_reward + np.random.normal(0, 0.05)
            elif etype == 1: # Contextual Liar
                if np.dot(self.lie_vectors[i], context) > 0:
                    fb = (1.0 - true_reward) + np.random.normal(0, 0.05) 
                else:
                    fb = true_reward + np.random.normal(0, 0.05) 
            elif etype == 2: # Sycophant
                fb = 0.95 + np.random.normal(0, 0.05)
            feedback.append(fb)
        return np.array(feedback)

# ==========================================
# 3. EXPERIMENT RUNNER
# ==========================================

def run_single_config(T, D, M, K, axiom_prob=0.15, seed=42):
    np.random.seed(seed)
    env = ScalableJekyllHydeEnvironment(D, K, M)
    esa = ESALinUCB(K, D, M, alpha=0.5, lambda_reg=0.1, axiom_prob=axiom_prob)
    naive = StandardLinUCB(K, D, alpha=0.5, lambda_reg=0.1)
    
    metrics = {'regret_esa': [], 'regret_naive': []}
    cum_esa = 0
    cum_naive = 0
    
    for t in range(T):
        ctx = env.get_context()
        true_rewards = [env.get_true_reward(ctx, a) for a in range(K)]
        opt_reward = max(true_rewards)
        
        # ESA
        act = esa.select_action(ctx)
        outcome = env.get_true_reward(ctx, act)
        fb = env.get_social_feedback(ctx, act, outcome)
        esa.run_step(ctx, act, fb, lambda c, a: env.get_true_reward(c, a))
        cum_esa += (opt_reward - outcome)
        
        # Naive
        act_n = naive.select_action(ctx)
        outcome_n = env.get_true_reward(ctx, act_n)
        fb_n = env.get_social_feedback(ctx, act_n, outcome_n)
        naive.run_step(ctx, act_n, fb_n)
        cum_naive += (opt_reward - outcome_n)
        
        metrics['regret_esa'].append(cum_esa)
        metrics['regret_naive'].append(cum_naive)
        
    return metrics, esa.log_trust_weights

if __name__ == "__main__":
    setup_plot_style()
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: The "Benchmark" (Hostile Majority)
    # T=10,000 for convergence check
    # ---------------------------------------------------------
    T_LONG = 10000
    print(f"Generating Figure 1 (Benchmark: D=20, M=10, T={T_LONG})...")
    res1, _ = run_single_config(T=T_LONG, D=20, M=10, K=5)
    
    plt.figure()
    plt.plot(res1['regret_naive'], label='Standard LinUCB', color=GRAY_BASELINE, linestyle='--')
    plt.plot(res1['regret_esa'], label='ESA-LinUCB (Ours)', color=RLJ_GREEN, linewidth=2.5)
    plt.title("Failure of Consensus (Hostile Majority)")
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Latent Regret")
    plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig("Figure_1_Regret.pdf") # Save as PDF for LaTeX
    plt.savefig("Figure_1_Regret.png", dpi=300) # Save as PNG for preview
    plt.close()

    # ---------------------------------------------------------
    # EXPERIMENT 2: Trust Dynamics (Visual Demo)
    # ---------------------------------------------------------
    print("Generating Figure 2 (Visual Demo: D=5, M=3)...")
    _, trust_logs = run_single_config(T=T_LONG, D=5, M=3, K=3)
    
    plt.figure()
    weights = np.array(trust_logs)
    window = 100 # Smoothing window for cleaner plot
    smooth_w = np.zeros_like(weights)
    for i in range(3):
        padded = np.pad(weights[:, i], (window//2, window//2), mode='edge')
        smooth_w[:, i] = np.convolve(padded, np.ones(window)/window, mode='valid')[:len(weights)]
        
    plt.plot(smooth_w[:, 0], label='Honest (Type 0)', color=HONEST_GREEN)
    plt.plot(smooth_w[:, 1], label='Contextual (Type 1)', color=CONTEXTUAL_ORANGE, linestyle='--')
    plt.plot(smooth_w[:, 2], label='Sycophant (Type 2)', color=SYCOPHANT_RED, linestyle=':')
    plt.title("Internal Trust Dynamics")
    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Trust Weight")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig("Figure_2_Trust.pdf")
    plt.savefig("Figure_2_Trust.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # EXPERIMENT 3: Scalability Sweep
    # ---------------------------------------------------------
    print("Generating Figure 3 (Scalability Sweep)...")
    dims = [10, 20, 50, 100]
    final_regrets = []
    
    for d in tqdm(dims):
        res, _ = run_single_config(T=5000, D=d, M=10, K=5)
        final_regrets.append(res['regret_esa'][-1])
        
    plt.figure()
    plt.plot(dims, final_regrets, marker='o', markersize=8, color=RLJ_GREEN, linewidth=2.5)
    plt.title("Scalability (Regret vs Dimension)")
    plt.xlabel("Context Dimension (D)")
    plt.ylabel(f"Final Regret (T=5000)")
    plt.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Figure_3_Scalability.pdf")
    plt.savefig("Figure_3_Scalability.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # EXPERIMENT 4: Sensitivity to Axiom Cost (Axiom Probability)
    # ---------------------------------------------------------
    print("Generating Figure 4 (Sensitivity Analysis)...")
    axiom_probs = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 100]
    final_regrets_sensitivity = []
    
    # We use the Benchmark settings (D=20, M=10) but vary probability
    for p in tqdm(axiom_probs):
        res, _ = run_single_config(T=10000, D=20, M=10, K=5, axiom_prob=p)
        final_regrets_sensitivity.append(res['regret_esa'][-1])

    plt.figure()
    plt.plot(axiom_probs, final_regrets_sensitivity, marker='o', markersize=8, color=RLJ_GREEN, linewidth=2.5)
    plt.title("Sensitivity to Axiom Cost")
    plt.xlabel("Axiom Probability (p)")
    plt.ylabel("Final Regret")
    plt.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig("Figure_4_Sensitivity.pdf")
    plt.savefig("Figure_4_Sensitivity.png", dpi=300)
    plt.close()
    
    print("Sensitivity Analysis Complete.")
    
    print("Done. All figures generated in .pdf (for LaTeX) and .png (for viewing).")