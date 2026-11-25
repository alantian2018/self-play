"""Throwaway desktop viewer: python play_tic_tac_toe.py [--checkpoint path.pt]."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import torch

from experiments.tic_tac_toe import make_agent, make_env
from selfplay.checkpoints import latest_checkpoint


def load_agent(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    agent = make_agent()
    try:
        agent.load_state_dict(checkpoint.get("agent", checkpoint))
    except RuntimeError as error:
        raise ValueError("Checkpoint doesn't match the current tic-tac-toe Agent architecture.") from error
    return agent.eval().requires_grad_(False)


class Game:
    def __init__(self, agent, human_first=True):
        self.agent = agent
        self.env = make_env()
        self.env.reset()
        self.human = self.env.possible_agents[0 if human_first else 1]
        self.symbol = "X" if human_first else "O"
        self.agent_turn()

    @property
    def finished(self):
        return self.env.terminations[self.human] or self.env.truncations[self.human]

    def agent_turn(self):
        if not self.finished and self.env.agent_selection != self.human:
            obs, *_ = self.env.last()
            self.env.step(int(self.agent(obs)))

    def play(self, action):
        if self.finished or self.env.agent_selection != self.human:
            return False
        obs, *_ = self.env.last()
        if not 0 <= action < 9 or not obs["action_mask"][action]:
            return False
        self.env.step(action)
        self.agent_turn()
        return True

    def board(self):
        obs = self.env.observe(self.env.possible_agents[0])["observation"].reshape(9, 2)
        return ["X" if own else "O" if other else "" for own, other in obs]

    def status(self):
        if not self.finished:
            return f"Your turn — you are {self.symbol}"
        reward = self.env.rewards[self.human]
        return "You win!" if reward > 0 else "Checkpoint wins." if reward < 0 else "Draw."

    def close(self):
        self.env.close()


class App:
    def __init__(self, root, checkpoint):
        self.root, self.agent, self.game = root, None, None
        root.title("Tic-tac-toe vs checkpoint")
        root.resizable(False, False)
        self.first = tk.BooleanVar(value=True)
        self.path = tk.StringVar(value="Choose a checkpoint to start")
        self.status = tk.StringVar(value="No checkpoint loaded")
        tk.Label(root, text="Tic-tac-toe", font=("Helvetica", 22, "bold")).pack(pady=(16, 8))
        tk.Label(root, textvariable=self.path, wraplength=390).pack(padx=16)
        controls = tk.Frame(root)
        controls.pack(pady=10)
        tk.Button(controls, text="Load checkpoint…", command=self.choose).pack(side="left", padx=4)
        tk.Button(controls, text="New game", command=self.new_game).pack(side="left", padx=4)
        tk.Checkbutton(root, text="Play first (X)", variable=self.first, command=self.new_game).pack()
        board = tk.Frame(root)
        board.pack(padx=20, pady=12)
        self.buttons = {}
        for row in range(3):
            for col in range(3):
                action = col * 3 + row  # Match PettingZoo's displayed cell order.
                button = tk.Button(board, width=3, height=1, font=("Helvetica", 48),
                                   command=lambda a=action: self.play(a))
                button.grid(row=row, column=col, padx=3, pady=3)
                self.buttons[action] = button
        tk.Label(root, textvariable=self.status, font=("Helvetica", 15)).pack(pady=(0, 16))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        if checkpoint:
            self.load(checkpoint)

    def choose(self):
        path = filedialog.askopenfilename(
            initialdir="release_checkpoints/tic_tac_toe",
            filetypes=[("PyTorch checkpoints", "*.pt *.pth"), ("All files", "*")],
        )
        if path:
            self.load(path)

    def load(self, path):
        try:
            agent = load_agent(path)
        except (OSError, RuntimeError, ValueError) as error:
            messagebox.showerror("Couldn't load checkpoint", str(error))
            return
        self.agent = agent
        self.path.set(str(path))
        self.new_game()

    def new_game(self):
        if self.agent is None:
            return
        if self.game:
            self.game.close()
        self.game = Game(self.agent, self.first.get())
        self.refresh()

    def play(self, action):
        if self.game:
            self.game.play(action)
            self.refresh()

    def refresh(self):
        marks = self.game.board() if self.game else [""] * 9
        for action, button in self.buttons.items():
            playable = self.game is not None and not self.game.finished and not marks[action]
            button.config(text=marks[action], state="normal" if playable else "disabled")
        if self.game:
            self.status.set(self.game.status())

    def close(self):
        if self.game:
            self.game.close()
        self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = latest_checkpoint("tic_tac_toe")
    torch.set_num_threads(1)
    root = tk.Tk()
    App(root, args.checkpoint)
    root.mainloop()
