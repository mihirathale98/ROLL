<div align="center">

<img src="assets/roll.jpeg" width="40%" alt="ROLL Logo">

# ROLL: Reinforcement Learning Optimization for Large-Scale Learning

<h4>🚀 An Efficient and User-Friendly Scaling Library for Reinforcement Learning with Large Language Models 🚀</h4>

<p>
  <a href="https://github.com/alibaba/ROLL/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/alibaba/ROLL/issues">
    <img src="https://img.shields.io/github/issues/alibaba/ROLL" alt="GitHub issues">
  </a>
  <a href="https://github.com/alibaba/ROLL/stargazers">
    <img src="https://img.shields.io/github/stars/alibaba/ROLL?style=social" alt="Repo stars">
  </a>
  <a href="https://arxiv.org/abs/2506.06122"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>
  <!-- 组织主页：点击跳转到 https://github.com/alibaba -->
  <a href="./assets/roll_wechat.png" target="_blank">
    <img src="https://img.shields.io/badge/WeChat-green?logo=wechat" alt="WeChat QR">
  </a>
</p>

</div>

ROLL is an efficient and user-friendly RL library designed for Large Language Models (LLMs) utilizing Large Scale GPU resources. It significantly enhances LLM performance in key areas such as human preference alignment, complex reasoning, and multi-turn agentic interaction scenarios.

Leveraging a multi-role distributed architecture with Ray for flexible resource allocation and heterogeneous task scheduling, ROLL integrates cutting-edge technologies like Megatron-Core, SGLang and vLLM to accelerate model training and inference.



---

## 📢 News

| 📣   Updates                                                                                                                                                                                                                                                                                              |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **[08/13/2025]** 🎉 ROLL supports AMD GPUs with out-of-box image docker and Dockerfile and specific yamls under `examples/` directory. Please refer to [Installation](https://alibaba.github.io/ROLL/docs/English/QuickStart/installation).                                                               |
| **[08/11/2025]** 🎉 Our Paper released, see [Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning](https://arxiv.org/abs/2508.08221).                                                                                                                                                           |
| **[08/10/2025]** 🎉 Agentic RL supports [stepwise learning](examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_gigpo.yaml), like [GiGPO](https://arxiv.org/abs/2505.10978); Distill supports [VLM](examples/qwen2.5-vl-7B-distill/distill_vl_megatron.yaml). Explore the new capabilities!               |
| **[07/31/2025]** 🎉 Refactor agentic rl design. Support agentic rl [async training](examples/qwen2.5-0.5B-agentic/agent_val_frozen_lake_async.yaml). Explore the new capabilities!                                                                                                                        |
| **[07/31/2025]** 🎉 Support [DistillPipeline](examples/qwen2.5-7B-distill_megatron/run_distill_pipeline.sh)/[DpoPipeline](examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh). Support [lora](examples/qwen2.5-7B-rlvr_megatron/rlvr_lora_zero3.yaml). Support [GSPO](https://arxiv.org/abs/2507.18071) |
| **[06/25/2025]** 🎉 Support thread env for env scaling and support qwen2.5 VL agentic pipeline.                                                                                                                                                                                                           |
| **[06/13/2025]** 🎉 Support Qwen2.5 VL rlvr pipeline and upgrade mcore to 0.12 version.                                                                                                                                                                                                                   |
| **[06/09/2025]** 🎉 ROLL tech report is now available! Access the report [here](https://arxiv.org/abs/2506.06122).                                                                                                                                                                                        |
| **[06/08/2025]** 🎉Supports  Qwen3(8B/14B/32B), Qwen3-MoE(30A3/235A22), Qwen2.5(7B/14B/32B/72B) LLM models.                                                                                                                                                                                          |
| **[05/30/2025]** 🎉 Training RLVR and Agentic RL with ROLL is now available! Explore the new capabilities.                                                                                                                                                                                                |
---


## 🚀 Get Started

[Documents](https://alibaba.github.io/ROLL/)

### Quick Start
[Installation](https://alibaba.github.io/ROLL/docs/English/QuickStart/installation)  
[Config System Explanation](https://alibaba.github.io/ROLL/docs/English/QuickStart/config_system)  
[Debugging Guide](https://alibaba.github.io/ROLL/docs/English/QuickStart/debugging_guide_en)  
[Trackers and Metrics](https://alibaba.github.io/ROLL/docs/English/UserGuide/trackers_and_metrics)  
[Checkpoint Saving and Resuming Guide](https://alibaba.github.io/ROLL/docs/English/UserGuide/checkpoint_and_resume)  
[Converting MCoreAdapter Models to Hugging Face Format](https://alibaba.github.io/ROLL/docs/English/UserGuide/megatron_convert_2_hf)  
[Quick Start: Single-Node Deployment Guide](https://alibaba.github.io/ROLL/docs/English/QuickStart/single_node_quick_start)  
[Quick Start: Multi-Node Deployment Guide](https://alibaba.github.io/ROLL/docs/English/QuickStart/multi_node_quick_start)  
[Frequently Asked Questions](https://alibaba.github.io/ROLL/docs/English/QuickStart/qa_issues)

### UserGuide

#### Pipeline Step by Step
[RLVR Pipeline](https://alibaba.github.io/ROLL/docs/English/UserGuide/pipeline/rlvr_pipeline_start)  
[Agentic Pipeline](https://alibaba.github.io/ROLL/docs/English/UserGuide/pipeline/agentic_pipeline_start)  
[Agentic Comprehensive Guide](https://alibaba.github.io/ROLL/docs/English/UserGuide/pipeline/agent_pipeline_start)  
[Distill Pipeline](https://alibaba.github.io/ROLL/docs/English/UserGuide/pipeline/distill_pipeline_start)

#### Algorithms
[Reinforce++](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/Reinforce_Plus_Plus)  
[TOPR](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/TOPR)  
[GiGPO](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/agentic_GiGPO)  
[PPO](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/PPO)  
[Lite PPO](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/LitePPO)  
[GRPO](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/GRPO)  
[GSPO](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/GSPO)  
[RAFT++](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/RAFT_Plus_Plus)  
[StarPO](https://alibaba.github.io/ROLL/docs/English/UserGuide/algorithms/agentic_StarPO)

#### Beckend
[DeepSeed](https://alibaba.github.io/ROLL/docs/English/UserGuide/backend/deepspeed)  
[Megatron](https://alibaba.github.io/ROLL/docs/English/UserGuide/backend/megatron)  
[LoRA](https://alibaba.github.io/ROLL/docs/English/UserGuide/backend/lora)  
[vLLM](https://alibaba.github.io/ROLL/docs/English/UserGuide/backend/vllm)  
[SGLang](https://alibaba.github.io/ROLL/docs/English/UserGuide/backend/sglang)

#### Advanced Features
[Agentic Asynchronous Parallel Rollout](https://alibaba.github.io/ROLL/docs/English/UserGuide/agentic_async_parallel_rollout)  
[Agentic Asynchronous Training Feature](https://alibaba.github.io/ROLL/docs/English/UserGuide/async_training_agentic)  

#### Performance Optimization & Resource Management 
[Resource Config](https://alibaba.github.io/ROLL/docs/English/UserGuide/device_mapping)   
[GPU Time-Division Multiplexing Control](https://alibaba.github.io/ROLL/docs/English/UserGuide/offload_reload_control)  


---

## ✨ Key Features
*   **Multi-task RL Training (RLVR):** Covers mathematics, coding, general reasoning, open-ended Q&A, instruction following, etc.
    *   Flexible `domain_batch_size` distribution control.
    *   **Sample-level asynchronous parallel Rollout**, asynchronous reward calculation, and dynamic sampling.
    *   Asynchronous training under implementation.
*   **Agentic RL:** Multi-turn interaction capabilities for games, multi-turn dialogues, tool use, etc.
    *   Environment-level **asynchronous parallel rollout**.
    *   Supports **asynchronous training**.
    *   Multi-turn interaction rollout supports **local debugging**, improving multi-turn interaction business development efficiency.
    *   Supports **TrajectoryWise (StartPO)** and **StepWise (GiGPO)** training paradigms.
*   **Algorithm-Friendly:** Provides flexible and rich RL strategy configurations by default.
    *   Over 20 rich reinforcement learning strategy options, such as reward normalization, reward clipping, various advantage estimation methods, etc.
    *   Out-of-the-box support for reinforcement learning algorithms, such as **PPO, GRPO, Reinforce++, TOPR, RAFT++, GSPO**, etc.
*   **Rich Training and Inference Engine:** Ray-based multi-role distributed architecture; Strategy abstraction unifies various backends, enabling easy operation from single machines to thousands-of-GPU clusters.
    *   Inference/Generation supports vLLM, SGLang.
    *   Training supports DeepSpeed (ZeRO), Megatron-LM 5D parallelism (mcore-adapter, dp/tp/pp/cp/ep), FSDP under implementation.
    *   Extreme offload/reload capabilities.
    *   Supports LoRA training.
    *   Supports FP8 rollout (FP8 inference for LLM as judge, FP8 rollout with BF16 training under development).
*   **AutoDeviceMapping:** Supports custom device mapping for different roles, flexibly managing colocated and disaggregated deployments.
*   **Observability:** Integrated with SwanLab / WandB / TensorBoard, tracking of performance for each domain and reward type.
*   **Rich Post-training Technical Support:**
    *   Agentic RL LLM & VLM
    *   RLVR LLM & VLM
    *   Distill Pipeline LLM & VLM
    *   DPO Pipeline
    *   SFT Pipeline under development



---

## 🔮 Upcoming Features

We are continuously working to expand ROLL's capabilities:
* ⏱️ **Async RLVR pipeline**: For even more efficient and streamlined asynchronous operations.
* ⚙️ **FSDP2**: Integrating the latest Fully Sharded Data Parallel techniques.
* 🔍 **Support DeepseekV3**: Adding compatibility for the newest Deepseek models.

---

## 🏆 Notable work based on ROLL
- [RecGPT](https://www.arxiv.org/abs/2507.22879): a next-generation, LLM-driven framework that places user intent at the core of recommender systems, fostering a more sustainable and mutually beneficial ecosystem.

-----

## 🙏 Citation and Acknowledgement

ROLL is inspired by the design of OpenRLHF, VeRL, Nemo-Aligner, and RAGEN.
The project is developed by Alibaba TAOBAO & TMALL Group and Alibaba Group. The code is distributed under the Apache License (Version 2.0). This product contains various third-party components under other open-source licenses. See the `NOTICE` file for more information.

The following repositories have been used in ROLL, either in their close-to-original form or as an inspiration:

  * [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  * [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
  * [sgl-project/sglang](https://github.com/sgl-project/sglang)
  * [vllm-project/vllm](https://github.com/vllm-project/vllm)

If you use ROLL in your research or project, please consider citing us:

```bibtex
@article{wang2025reinforcement,
  title={Reinforcement Learning Optimization for Large-Scale Learning: An Efficient and User-Friendly Scaling Library},
  author={Wang, Weixun and Xiong, Shaopan and Chen, Gengru and Gao, Wei and Guo, Sheng and He, Yancheng and Huang, Ju and Liu, Jiaheng and Li, Zhendong and Li, Xiaoyang and others},
  journal={arXiv preprint arXiv:2506.06122},
  year={2025}
}
```



-----

## 🤝 About [ROLL Team]
ROLL is a project jointly developed by Taotian Future Life Lab and Aicheng Technology, with a strong emphasis on pioneering the future of Reinforcement Learning (RL). Our mission is to explore and shape innovative forms of future living powered by advanced RL technologies. If you are passionate about the future of RL and want to be part of its evolution, we warmly welcome you to join us! Learn more about the ROLL Team through our official channels below👇

<a href="./assets/roll_wechat.png" target="_blank">
  <img src="https://img.shields.io/badge/WeChat-green?logo=wechat" alt="WeChat QR">
</a>

-----
We are HIRING! 
- Post Training Infra 研发工程师 [JD link](https://talent-holding.alibaba.com/off-campus/position-detail?lang=zh&positionId=7000016304)
- 大模型训练专家： 
  - （社招）[JD link](https://talent.taotian.com/off-campus/position-detail?lang=zh&positionId=7000024203)
  - （校招）[JD link](https://talent.taotian.com/campus/position-detail?positionId=199900140053)
- Infra 研究型实习生 [JD link](https://talent-holding.alibaba.com/campus/position-detail?lang=zh&positionId=59900004115)

-----

<div align="center">
We welcome contributions from the community! 🤝
</div>
