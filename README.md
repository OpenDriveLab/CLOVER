<h1 align="center"> :four_leaf_clover: CLOVER: Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation </h1> 

<div id="top" align="center">
<p align="center">
<img src="assets/clover_teaser.png" width="1000px" >
</p>
</div>

> [Qingwen Bu](https://scholar.google.com/citations?user=-JCRysgAAAAJ&hl=zh-CN&oi=ao), [Jia Zeng](https://scholar.google.com/citations?hl=zh-CN&user=kYrUfMoAAAAJ), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=zh-CN), Yanchao Yang, Guyue Zhou, Junchi Yan, Ping Luo, Heming Cui, Yi Ma and [Hongyang Li](https://lihongyang.info/)
>

Code is coming soon. Please stay tuned.🦾

## :fire: Highlight

:four_leaf_clover: ​**CLOVER**  employs a text-conditioned video diffusion model for generating visual plans as reference inputs, then these sub-goals guide the feedback-driven policy to generate actions with an error measurement strategy.

<div id="top" align="center">
<p align="center">
<img src="assets/closed-loop.jpg" width="1000px" >
</p>
</div>

Owing to the closed-loop attribute, ​**CLOVER** is robust to visual distraction and object variation:
<div id="top" align="center">
<p align="center">
<img src="assets/vis_robustness.jpg" width="1000px" >
</p>
</div>

This closed-loop mechanism enables achieving the desired states accurately and reliably, thereby facilitating the execution of long-term tasks:
<div id="top" align="center">
<p align="center">
<img src="assets/long-horizon-task.gif" width="1000px" >
</p>
</div>

<!-- ## :loudspeaker: News

- **[2024/09/xx]** We released our paper on arXiv. -->

## :pushpin: TODO list

- [ ] Training script for visual planner
- [ ] Training script for feedback-driven policy
- [ ] Evaluation codes on CALVIN
- [ ] Checkpoints release



<!-- ## :video_game: Getting Started

Generated videos conditioned on the same initial frame and differenct language instruction:
<div id="top" align="center">
<p align="center">
<img src="assets/gen_diff_condition.png" width="1000px" >
</p>
</div> -->

