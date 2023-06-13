# Reinforcement Learning and SOM for Grid and Country Clustering

This repository contains the implementation of a Reinforcement Learning (RL) algorithm for an n x n grid and a Self-Organizing Map (SOM) for clustering the "countries of the world.csv" dataset from Kaggle. The RL algorithm is applied to navigate an agent in a grid environment, while the SOM is used to cluster countries based on their attributes and visualize them on a world map and a grid representation.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [RL Algorithm Overview](#rl-algorithm-overview)
- [SOM for Country Clustering](#som-for-country-clustering)
- [Results](#results)
- [Contributors](#contributors)

## Introduction
In this assignment, I have implemented an RL algorithm for an n x n grid and a SOM for clustering countries based on their attributes. The RL algorithm allows an agent to learn optimal policies for navigating the grid environment, while the SOM provides a powerful visualization tool for grouping countries with similar attributes. The provided code showcases the implementation and application of these algorithms and can be further extended for more complex problems or datasets.

## Dependencies
The following dependencies are required to run the code:
- Python
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Geopandas (for world map visualization)
  
## RL Algorithm Overview
The implemented RL algorithm follows these main steps:

1. **Initialization**: Set up the grid environment, agent, and Q-table.
2. **Exploration and Exploitation**: Iteratively perform actions in the grid environment based on exploration and exploitation strategies.
3. **Action Selection**: Choose actions based on the current state and Q-values.
4. **Q-value Update**: Update the Q-table based on the observed rewards and new states.
5. Repeat steps 2-4 for a specified number of episodes or until convergence is reached.

The RL algorithm learns to navigate the grid environment by maximizing the cumulative rewards obtained from taking actions.

## SOM for Country Clustering
The Self-Organizing Map (SOM) is a neural network algorithm used for unsupervised learning and clustering. In this assignment, the SOM is applied to cluster countries based on their attributes from the "countries of the world.csv" dataset. The algorithm learns a low-dimensional representation of the countries and groups them based on similarities in their attribute vectors. The clustered countries are then visualized on a world map and a grid representation.

## Results
The repository includes example results obtained by running the RL algorithm for grid navigation and the SOM for country clustering and visualization. The results demonstrate the learned navigation policies in the grid environment and the clustered countries displayed on a world map and a grid representation.

## Contributors
This was a group assignment completed in collaboration with [Muhammad Murtaza]( https://github.com/mm06369/ ).

<a href="https://github.com/aliasgharchakera/CI-Spring23-Assignment03/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aliasgharchakera/CI-Spring23-Assignment03" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
