"""Desktop viewer: python play_connect4.py [--checkpoint path.pt]."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import torch

from experiments.connect4 import make_agent, make_env
from selfplay.checkpoints import latest_checkpoint


def load_agent(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    agent = make_agent()
    try:
        agent.load_state_dict(checkpoint.get("agent", checkpoint))
    except RuntimeError as error:
        raise ValueError("Checkpoint doesn't match the current Connect 4 Agent architecture.") from error
    return agent.eval().requires_grad_(False)


class Game:
    def __init__(self, agent, human_first=True):
        self.agent = agent
        self.env = make_env()
        self.env.reset()
        self.human = self.env.possible_agents[0 if human_first else 1]
        self.color = "red" if human_first else "black"
        self.agent_turn()

    @property
    def finished(self):
        return self.env.terminations[self.human] or self.env.truncations[self.human]

    def agent_turn(self):
        if not self.finished and self.env.agent_selection != self.human:
            obs, *_ = self.env.last()
            self.env.step(int(self.agent(obs)))

    def play(self, column):
        if self.finished or self.env.agent_selection != self.human:
            return False
        obs, *_ = self.env.last()
        if not 0 <= column < 7 or not obs["action_mask"][column]:
            return False
        self.env.step(column)
        self.agent_turn()
        return True

    def board(self):
        obs = self.env.observe(self.env.possible_agents[0])["observation"]
        return obs[..., 0] + 2 * obs[..., 1]

    def status(self):
        if not self.finished:
            return f"Your turn — you are {self.color}. Click a column."
        reward = self.env.rewards[self.human]
        return "You win!" if reward > 0 else "Checkpoint wins." if reward < 0 else "Draw."

    def close(self):
        self.env.close()


class App:
    CELL = 64

    def __init__(self, root, checkpoint):
        self.root, self.agent, self.game = root, None, None
        root.title("Connect 4 vs checkpoint")
        root.resizable(False, False)
        self.first = tk.BooleanVar(value=True)
        self.path = tk.StringVar(value="Choose a checkpoint to start")
        self.status = tk.StringVar(value="No checkpoint loaded")
        tk.Label(root, text="Connect 4", font=("Helvetica", 22, "bold")).pack(pady=(16, 8))
        tk.Label(root, textvariable=self.path, wraplength=440).pack(padx=16)
        controls = tk.Frame(root)
        controls.pack(pady=10)
        tk.Button(controls, text="Load checkpoint…", command=self.choose).pack(side="left", padx=4)
        tk.Button(controls, text="New game", command=self.new_game).pack(side="left", padx=4)
        tk.Checkbutton(root, text="Play first (red)", variable=self.first, command=self.new_game).pack()
        columns = tk.Frame(root)
        columns.pack(pady=(10, 2))
        self.buttons = []
        for col in range(7):
            columns.columnconfigure(col, minsize=self.CELL)
            button = tk.Button(columns, text=f"↓ {col + 1}", command=lambda c=col: self.play(c))
            button.grid(row=0, column=col, sticky="ew")
            self.buttons.append(button)
        self.canvas = tk.Canvas(root, width=7 * self.CELL, height=6 * self.CELL,
                                bg="#2354ae", highlightthickness=0)
        self.canvas.pack(padx=16)
        self.canvas.bind("<Button-1>", lambda event: self.play(event.x // self.CELL))
        tk.Label(root, textvariable=self.status, font=("Helvetica", 14)).pack(pady=16)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        if checkpoint:
            self.load(checkpoint)

    def choose(self):
        path = filedialog.askopenfilename(
            initialdir="release_checkpoints/connect4",
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

    def play(self, column):
        if self.game:
            self.game.play(column)
            self.refresh()

    def refresh(self):
        self.canvas.delete("all")
        board = self.game.board() if self.game else None
        colors = ("#f8f9fc", "#e74c3c", "#242424")
        for row in range(6):
            for col in range(7):
                mark = int(board[row, col]) if board is not None else 0
                x, y = col * self.CELL, row * self.CELL
                self.canvas.create_oval(x + 7, y + 7, x + self.CELL - 7, y + self.CELL - 7,
                                        fill=colors[mark], outline="")
        for col, button in enumerate(self.buttons):
            playable = self.game is not None and not self.game.finished and board[0, col] == 0
            button.config(state="normal" if playable else "disabled")
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
        args.checkpoint = latest_checkpoint("connect4")
    torch.set_num_threads(1)
    root = tk.Tk()
    App(root, args.checkpoint)
    root.mainloop()
