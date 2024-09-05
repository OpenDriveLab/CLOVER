<h1 align="center"> :four_leaf_clover: CLOVER: Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation </h1> 

<div id="top" align="center">
<p align="center">
<img src="assets/clover_teaser.png" width="1000px" >
</p>
</div>

## :fire: Highlight

:four_leaf_clover: ​**CLOVER**  closed-loop visuomotor control framework that incorporates feedback mechanisms to improve adaptive robotic control:

- We introduce CLOVER, a generalizable closed-loop visuomotor control framework that incorporates a feedback mechanism to improve adaptive robotic control.
- We investigate the state-measuring attribute for latent embeddings and propose a policy by quantifying feedback errors explicitly. The error quantification settles the construction of an execution pipeline to resolve the challenge of handling uncertainties in video generation and task horizons.
- Extensive experiments in simulation and real-world robots verify the effectiveness of CLOVER. It surpasses prior state-of-the-arts by a notable margin (+8%) on CALVIN. The average length of completed tasks on real-world long-horizon manipulation improves by 91%.

## :movie_camera: Demo
Long-horizon task execution:
<div id="top" align="center">
<p align="center">
<img src="assets/long-horizon-task.gif" width="1000px" >
</p>
</div>

Robustness to visual distractions and object variation:
<div id="top" align="center">
<p align="center">
<img src="assets/vis_robustness.jpg" width="1000px" >
</p>
</div>

Generated videos conditioned on the same initial frame and differenct language instruction:
<div id="top" align="center">
<p align="center">
<img src="assets/gen_diff_condition.png" width="1000px" >
</p>
</div>


## :loudspeaker: News

- **[2024/09/xx]** We released our paper on arXiv.

## :pushpin: TODO list

- [ ] Training script for visual planner
- [ ] Training script for feedback-driven policy
- [ ] Evaluation codes on CALVIN
- [ ] Checkpoints release



## :video_game: CALVIN Benchmark

