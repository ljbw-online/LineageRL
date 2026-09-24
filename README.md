# Lineage RL

In 2015 DeepMind demonstrated an AI agent that had achieved average human performance on a set
of Atari 2600 games. This agent was special because it didn't get any data directly from the
game, it didn't get access to the emulator, and it didn't get to see human demonstrations of
gameplay. It had learnt to play in the way a human does: by simply looking at the pixels of
the game window and deciding which actions to take.

This agent was called [Deep Q-networks](https://www.nature.com/articles/nature14236)
(DQN) and soon after its introduction a benchmark of 57 games was established as a way of
measuring an agent's performance. DQN could play the easiest games better than a human but
on the most difficult games it would completely fail to learn and would never achieve a
positive score.

It took another five years of work by the DeepMind researchers and their collaborators for
the most difficult games to be beaten. The result was Agent57, which was the first agent to
attain at least average human performance on all 57 games. In the process of creating it the
researchers had made inroads into some of the fundamental problems of reinforcement learning,
such as how to explore a very treacherous environment.

In the [blog post announcing Agent57](https://deepmind.google/blog/agent57-outperforming-the-human-atari-benchmark/)
 its predecessors were referred to as the *lineage* of Agent57.

This repository traces that lineage. I created it primarily to learn about the algorithms and
how they are implemented in code. But I also hope that it will be useful to other people for that
purpose as well, and that it will be useful for composing new agents from modular components.

This repository is not yet in a useable state but I will add usage instructions as soon as it
is.
