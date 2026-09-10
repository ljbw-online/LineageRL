# Lineage RL

In 2015 DeepMind demonstrated an AI agent that had achieved average human performance on a set
of Atari 2600 games. This agent was special because it didn't get any data directly from the
game, it didn't get access to the emulator, and it didn't get to see human demonstrations of
gameplay. It learnt played in the way a human does: by simply looking at the pixels of the game
window and deciding which actions to take.

Although this agent performed better than a human on certain games, on the most challenging
games it would completely fail to learn and would never achieve a positive score. It took
another five years of work by the DeepMind researchers and their collaborators to created an
agent which could learn to play these games. This agent was called Agent57 and was the first to
attain at least average human performance on all members of a benchmark of 57 Atari games. In
the [announcement blog post](https://deepmind.google/blog/agent57-outperforming-the-human-atari-benchmark/)
 its predecessors were referred to as the *lineage* of Agent57.

This repository traces that lineage. I created it primarily to learn about these algorithms and
how they are implemented in code. I also hope that it will be useful to other people for that
purpose as well, and that it will be useful for composing new agents from modular components.

This repository is not yet in a useable state but I will add usage instructions as soon as it
is.
