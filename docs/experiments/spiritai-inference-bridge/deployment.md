# SpiritAI 部署与故障处理

按机器角色恢复 Full Flow 真机推理环境，复用现有 SDK Docker image，并根据已发生的故障定位网络、协议和日志问题。[操作手册](README.md) · [参数说明](parameters.md) · [English](deployment.en.md)

## 有线网络与地址

当前工作站有线网口为 `enp44s31f6`，地址为 `172.16.0.50/24`；机器人 client 地址为 `172.16.0.30:8766`。Wi-Fi 保留互联网默认路由。有线链路曾停留在 DHCP 等待，未分配 IPv4，机器人流量因而被 Mihomo TUN 接管。

在工作站先检查实际路由：

```bash
ip -brief -4 address
ip -4 route get 172.16.0.30
```

预期包含 `dev enp44s31f6 src 172.16.0.50`。需要重配且确认 `.50` 未被其他设备占用时，再手动执行下面命令；网卡名、连接名和机器人子网以现场为准：

```bash
sudo nmcli connection modify "Wired connection 1" \
  ipv4.method manual ipv4.addresses "172.16.0.50/24" \
  ipv4.gateway "" ipv4.dns "" ipv4.never-default yes
sudo nmcli connection up "Wired connection 1" ifname enp44s31f6
ping -c 3 172.16.0.30
nc -vz -w 3 172.16.0.30 8766
```

地址属于同一子网且路由直达网口后，再排查端口。ping 通但 8766 不通时，检查 server 是否已监听、container 日志和防火墙，而不是修改模型权重。单节点 Ray 曾显示 `node_ip=198.18.0.1`，但机器人 socket 实际走有线 `.50`；仅凭该 Ray 信息不能判定机器人连接失败，也不表示推理使用两张 GPU。

## Docker 镜像与挂载源码

镜像由机器人本机 Docker 管理，tar 文件放置位置任意。`run_robot.sh` 使用镜像 tag，不会自动加载旁边的 tar；曾因镜像未导入/标签不匹配而尝试拉取不可达的 Harbor。

在机器人终端导入现有 tar 并核对 tag：

```bash
cd ~/robot_server_code
docker load -i thor-robot-0526.tar
docker image ls
```

期望 image 为 `harbor.i.spirit-ai.com:443/shenming/thor-robot:0526`；名称不同可通过 `IMAGE` 环境变量指定真实 tag。该镜像提供 ROS/SDK/相机依赖。目录应保持：

```text
~/robot_server_code/
├── run_robot.sh
├── thor-robot-0526.tar        # 导入后运行不依赖 tar 位置
└── robot_server/
    ├── main.py
    ├── server.py
    ├── mozrobot.py
    └── camera.py
```

脚本将旁边的 `robot_server/` bind-mount 到 `/root/robot_server`。因此 Python server 源码更新在宿主机目录完成，容器重启后生效；无需为这次兼容更新重建 `0526` image。`docker logs -f` 的 Ctrl+C 只停止查看日志；`docker stop robot_server` 才停止服务。反复执行 `run_robot.sh` 会删除并重建同名容器。

## 为什么原 OpenPI 可访问而严格检查失败

原 `spiritai_training_cart` 和 `rct-rebuild` bridge 都读取现有 `hello.metadata`，检查动作维度、相机和 `max_chunk` 等，没有要求 `execution_capabilities` 三项新字段。前者主要下发多行/prefix，后者明确下发单行。`rct-rebuild` 的 `rtc_capabilities / training_time_v1` 是 policy server 的模型能力声明，与机器人 server 的执行声明不同。

当前 RLinf 适配新增 `single_step_cartesian`、`tracked_completion`、`cancel_actions` 作为执行检查。因此字段缺失只能证明“未声明”，不能单独证明原系统无法访问或执行。此次先确认只读连接正常，再通过机器人容器内 SHA256 证实运行源码与已审查原版一致，之后才应用对应修复。

| 文件 | 原始 SHA256 | 兼容 SHA256 |
|---|---|---|
| `server.py` | `2fca22e0cffe72773858049f2ece43627faf94394612641c60f35255c6da0c7f` | `5fc600426233c4b767016cf10596cb9b3189093d19b77dbeb6fa6babe656f2f8` |
| `mozrobot.py` | `464e3d8f1db9f2c0374cbd88a4fda2ae79767aef7389a927c08fee50e3ffce17` | `de7f93ef36f80b47984000fb33d2ac65eda427bdba527dbc7009c270aef9a7d8` |

这份原始源码的单行 `resample(T=1)` 会进入 PCHIP/SLERP 并报错；busy 依预计时间清除；`cancel_actions()` 为空。兼容补丁补齐单行路径、后台 Future/错误跟踪，以及停止后续 Python SDK 提交的取消协议。Future 完成表示下发任务结束，不代表机器人已到达目标；已交给 SDK/控制器的目标也不会因软件取消自动撤销。

## 源码包、备份与重启

2026-11-19 已由用户在机器人手动部署并通过严格 preflight，版本为 `robot-server/0.3.1-rlinf-compat`。工作站原始 `/home/deng/Documents/codebase/robot_server_code/` 保留原版；兼容副本位于 `/home/deng/Documents/codebase/robot_server_code_rlinf_compat_20261119/`，其中的 `rlinf_server_source_compat_20261119.tar.gz` 包含两个源码文件、before/after 校验清单与 manifest。它是源码包，不是 Docker image。

需要恢复本次部署流程时，在工作站上传已有包：

```bash
scp /home/deng/Documents/codebase/robot_server_code_rlinf_compat_20261119/rlinf_server_source_compat_20261119.tar.gz \
  spirit-ai@172.16.0.30:~/robot_server_code/
```

随后在机器人上校验并部署。原版 before 校验只适用于尚未更新的源码；已经部署的机器应核对 after，不要再次强行套用原版检查：

```bash
cd ~/robot_server_code
mkdir -p rlinf-compat-20261119
tar -xzf rlinf_server_source_compat_20261119.tar.gz -C rlinf-compat-20261119
sha256sum -c rlinf-compat-20261119/before.sha256 &&
docker stop robot_server &&
cp -a robot_server "robot_server.backup.$(date +%Y%m%d_%H%M%S)" &&
cp rlinf-compat-20261119/robot_server/server.py \
   rlinf-compat-20261119/robot_server/mozrobot.py robot_server/ &&
sha256sum -c rlinf-compat-20261119/after.sha256 &&
bash run_robot.sh
```

命令仅在前一步成功时继续。修改源码需要停止并重启服务；读取指纹则可在服务运行时新开 session 执行：

```bash
docker exec robot_server sha256sum \
  /root/robot_server/server.py /root/robot_server/mozrobot.py
```

副本或源码包丢失时，可从支持的原版重新生成；下列命令要求全新的输出目录，当前已有目录不要重复生成：

```bash
.venv/bin/python -m toolkits.realworld_check.prepare_spiritai_server \
  --source-root /home/deng/Documents/codebase/robot_server_code \
  --output /path/to/new_compat_copy
```

工具核对精确原版 hash、复制源码、应用归档补丁并核对兼容 hash，生成 `robot_server/` 与 `compatibility.json`。原始[补丁](../pi05-explicit-subtask-ce-lora/artifacts/robot_server_compat.patch)和[版本清单](../pi05-explicit-subtask-ce-lora/artifacts/robot_server_compat_manifest.json)保存在 Git 文档中，不依赖 `/tmp` 存活。打包时从 manifest 的 `source_sha256` / `patched_sha256` 生成 `before.sha256` / `after.sha256`，再只打包两个源码、两份校验清单和 manifest；不覆盖 camera 或 main。

启动后在工作站运行 README 的严格 preflight。它不发动作，不改变 following。`external_following_enabled=true` 表示此前已经启用，不是 preflight 启用；`actions_sent=0` 只说明这次检查没有发送动作。能力声明通过后，仍以现场短测确认 SDK 单位、运动方向和控制器响应。

## W&B 与权重加载错误的区分

本次第一次推理失败在 `EmbodiedEvalRunner.__init__ → MetricLogger → wandb.init`，此时尚未调用 `runner.init_workers()`，权重没有开始加载。根因与先前训练记录相同：凭据绑定 `api.forge.coreweave.com`，新进程却默认访问 `api.wandb.ai`。

推理与 RTC 不依赖 W&B。README 的命令用 `runner.logger.logger_backends=[]` 关闭云端指标，保留终端和 RTC JSONL。需要云端记录时，删除该覆盖并在同一启动环境中设置：

```bash
export WANDB_BASE_URL=https://api.forge.coreweave.com
```

后台 CLI/服务运行与当前 Python 进程的 endpoint 配置分别管理。无需把密钥写入 YAML、命令或文档。仅在确认 endpoint 匹配之后，才进一步排查认证；不要把所有 `Error executing job` 都归类为权重损坏。

## 按阶段定位故障

先看失败处于启动、推理、RPC 还是收尾阶段，再使用下表。保存本轮解析后的配置、trace 和现场现象，避免改变多个参数后丢失对照。

| 现象 | 依据与处理 |
|---|---|
| Harbor 超时 / image not found | 先 `docker load` 并匹配脚本 tag；tar 放在目录中不等于已导入 |
| listening 尚未出现 | SDK/ROS2 或相机可能仍在初始化；看 container 日志，不能只看 Docker Up |
| connected 随即 disconnected | 对照客户端异常；此前是 hello 后严格字段检查失败，尚未读 obs 或执行模型 |
| missing execution capabilities | 先做不带 `--require-execution` 的只读检查，再核对实际源码；不能仅靠添加 flags 或绕过检查 |
| 权重加载几十秒无动作 | 先看 hash/加载阶段；profile 预热后才执行，不能把加载等待当作控制延迟 |
| 多张 GPU placement 打印 | 框架枚举全机资源；看实际 rollout placement 的唯一 rank 和 GPU 进程，不能按日志行数推断模型副本数 |
| `Evaluating RTC Rollout: Nit` | 这是请求数；实际动作 tick 数看 `episode_len` 和 trace 的 `step` |
| replan 超时 / exhausted chunk | 对照 `request_seconds`、`inference_seconds`、RPC 和动作索引；在剩余 horizon 内恢复，不能无限复用末目标 |
| `reward=0 / success=0` | 当前环境没有自动折盒成功判定；结合动作、trace 和人工任务结果评价 |
| `close_error` / following 未释放 | 检查取消 ACK、SDK 下发错误和 mode 回复；软件取消不等于硬件急停 |

Ctrl+C 走软件停止与关闭流程；硬杀进程或节点故障依赖 server 的断连处理。恢复时先确认无旧客户端占用，再重置现场构型并重新 bootstrap。当前 Full Flow 真机结果见[验证记录](implementation.md)，参数旋钮见[参数说明](parameters.md)。
