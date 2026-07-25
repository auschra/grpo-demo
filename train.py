from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify
import numpy as np
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Hyperparams
model_name = "Qwen/Qwen3-0.6B"
group_size = 2                           # number of generations in each group
input_batch_size = 2                      # number of input questions to process at once
epsilon = 0.2
beta = 0.04
lr = 1e-3
weight_decay = 1e-1
warmup_steps = 2000
max_steps = 300



def collate_fn(batch):
    questions = [example["problem"] for example in batch]
    answers = [example["solution"] for example in batch]

    return questions, answers

# setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# setup policy and frozen reference model
policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
reference_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
for param in reference_model.parameters():
    param.requires_grad = False
reference_model.eval()

ds = load_from_disk('./local_numinamath')
dataloader = DataLoader(ds, batch_size=input_batch_size, shuffle=True, collate_fn=collate_fn)
optimizer = AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)

def calculate_reward(generation, gold):
    """
    Check the ground truth label matches the generation for batch, assign 1.0 reward if so, 0.0 if not
    return:
        list of rewards for single group (each problem)
    """

    group_rewards = []

    # Parse true answer first
    gold_parse = parse(gold)

    for generation in generation, gold:
        answer_parse = parse(generation)
        is_correct = verify(gold_parse, answer_parse)
        group_rewards.append(1.0 if is_correct else 0.0)

    return group_rewards

def train_step(prompts, true_answer):
    """
    Run forward pass for a single step, update weights with policy gradients
    """

    # Tokenise samples (prompt + answqer)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Generate answers [batch, seq length]
    with torch.no_grad():
        output_ids = policy_model.generate(
            **inputs,   
            max_new_tokens=200,                             # increase later
            do_sample=True,                                 # sample probabalistically
            num_return_sequences=group_size,                # number of seqs
            temperature=0.6,                                # high temp incr diversity
            pad_token_id=tokenizer.eos_token_id,             # pad with eos
            )

    # Remove prompt from output_ids: shape -> [batch, max_new_tokens ie seq_len]
    new_token_ids = output_ids[:, prompt_len:]


    # -- Verification reward --
    # Decode token ids into string for math-verify: shape -> [batch, (answer string)]
    answer_strings = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

    # Repeated to compare each: shape -> [ans1, ans2, ans3]
    group_gold = [ans for ans in true_answer for _ in range(group_size)]

    binary_rewards = calculate_reward(answer_strings, group_gold)
    rewards = torch.tensor(binary_rewards, dtype=torch.float32, device=device) 

    # -- Group advantage --
    rewards_reshaped = rewards.view(-1, group_size)
    group_mean = rewards_reshaped.mean(dim=1, keepdim=True)
    group_std = rewards_reshaped.std(dim=1, keepdim=True) + 1e-8      # avoid div 0
    advs = ((rewards_reshaped - group_mean) / group_std).view(-1)

    # Mask padding tokens
    att_mask = (output_ids != tokenizer.pad_token_id).long()

    # Get logits: shape -> [batch, seq_length, vocab_size]
    policy_logits = policy_model(output_ids, attention_mask=att_mask).logits
    with torch.no_grad():
        reference_logits = reference_model(output_ids, attention_mask=att_mask).logits

    # Drop logits for t+1 token: shape -> [batch, seq_length - 1, vocab_size]
    policy_logits = policy_logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    # Shift out_ids by 1 to get next token label algined: shape -> [batch, seq_length - 1]
    label_ids = output_ids[:, 1:]

    # Shift attention to align with logits 
    logit_mask = att_mask[:, 1:]

    # Softmax to get log probs: shape -> [batch, seq_length - 1, vocab_size]
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)

    # Get log probs of generated tokens only: shape -> [batch, seq_length - 1]
    policy_log_probs_token = torch.gather(policy_log_probs, dim = -1, index=label_ids.unsqueeze(-1)).squeeze(-1)
    reference_log_probs_token = torch.gather(reference_log_probs, dim = -1, index=label_ids.unsqueeze(-1)).squeeze(-1)

    # Mask out input tokens
    generated_mask = torch.zeros_like(logit_mask)        # create zero mask
    generated_mask[:, prompt_len-1:] = 1                  # apply to input toks (1 to new)
    answer_mask = logit_mask * generated_mask            # new toks without padding toks

    # -- KL Divergence penalty --
    log_prob_diff = reference_log_probs_token - policy_log_probs_token
    kl_div = torch.exp(log_prob_diff) - log_prob_diff - 1

    # -- Clip reward --
    # call log probs ratio
    ratio = torch.exp(policy_log_probs_token - policy_log_probs_token.detach())     # detach to keep old policy for next step
    expanded_advs = advs.unsqueeze(-1).expand_as(ratio)                        # 1D adv scores stretched to cover all toks

    unclipped_reward = ratio * expanded_advs
    clipped_reward = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * expanded_advs   
    
    # -- Calc loss --
    token_loss = -torch.min(unclipped_reward, clipped_reward) + beta * kl_div       # convert reward into loss for each token
    loss = (token_loss * answer_mask).sum() / answer_mask.sum()                     # average over sequence

    # -- Update policy --
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.2f}, mean reward: {group_mean.item():.2f}")



print("Training start")
global_step = 0

while global_step < max_steps:
    for prompts, true_answer in dataloader:
        if global_step >= max_steps:
            break
            
        train_step(prompts, true_answer)
        
        global_step += 1
        if global_step % 10 == 0:
            print(f"Completed {global_step}/{max_steps} steps.")

print("Training finished")