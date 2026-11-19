# SpiritAI Deployment and Troubleshooting

Restore the Full Flow robot environment by machine role, reuse the SDK Docker image, and diagnose the networking, protocol, and logging failures encountered during deployment. [Runbook](README.en.md) · [Parameters](parameters.en.md) · [中文](deployment.md)

## Wired Network and Addresses

The workstation uses `enp44s31f6` at `172.16.0.50/24`; the robot endpoint is `172.16.0.30:8766`. Wi-Fi retains the Internet default route. Initially the Ethernet profile waited for DHCP without an IPv4 address, so Mihomo TUN captured robot traffic.

Check the workstation route first:

```bash
ip -brief -4 address
ip -4 route get 172.16.0.30
```

Expect `dev enp44s31f6 src 172.16.0.50`. If reconfiguration is needed and `.50` is unused by other devices, run manually, adapting interface/profile names and subnet to the site:

```bash
sudo nmcli connection modify "Wired connection 1" \
  ipv4.method manual ipv4.addresses "172.16.0.50/24" \
  ipv4.gateway "" ipv4.dns "" ipv4.never-default yes
sudo nmcli connection up "Wired connection 1" ifname enp44s31f6
ping -c 3 172.16.0.30
nc -vz -w 3 172.16.0.30 8766
```

Once the same-subnet direct route works, diagnose the port. If ping works but 8766 does not, inspect listening state, container logs, and firewall rather than weights. Single-node Ray reported `node_ip=198.18.0.1` while the robot socket used Ethernet `.50`; that Ray field alone does not establish a robot connection failure or two-GPU inference.

## Docker Image and Mounted Source

The robot's Docker daemon manages images; an image tar can be anywhere. `run_robot.sh` selects a tag and does not load an adjacent tar automatically. An absent/mismatched local image caused attempts to pull from unreachable Harbor.

Load the existing tar and inspect its tag on the robot:

```bash
cd ~/robot_server_code
docker load -i thor-robot-0526.tar
docker image ls
```

Expect `harbor.i.spirit-ai.com:443/shenming/thor-robot:0526`; use the `IMAGE` environment variable for a different loaded tag. It supplies ROS/SDK/camera dependencies. Retain this source layout:

```text
~/robot_server_code/
├── run_robot.sh
├── thor-robot-0526.tar        # runtime uses the imported image
└── robot_server/
    ├── main.py
    ├── server.py
    ├── mozrobot.py
    └── camera.py
```

The launcher bind-mounts adjacent `robot_server/` at `/root/robot_server`. Update Python source on the host and restart the container; this compatibility change reuses the `0526` image. Ctrl+C in `docker logs -f` only stops the viewer, while `docker stop robot_server` stops the service. Rerunning `run_robot.sh` removes and recreates the named container.

## Why Original OpenPI Connected but Strict Preflight Failed

Both original `spiritai_training_cart` and `rct-rebuild` bridges use existing `hello.metadata` fields such as dimensions, cameras and `max_chunk`, without the three `execution_capabilities` fields. The former primarily submits multiple rows/prefixes, while the latter explicitly submits one row. Its `rtc_capabilities / training_time_v1` describe the policy server and are separate from robot execution declarations.

This RLinf adapter added `single_step_cartesian`, `tracked_completion`, and `cancel_actions` as deployment checks. Missing fields prove no declaration, not that the old robot system cannot connect or act. Here, read-only access passed first; SHA256 from the actual container then matched reviewed original source before a targeted patch was applied.

| File | Original SHA256 | Compatible SHA256 |
|---|---|---|
| `server.py` | `2fca22e0cffe72773858049f2ece43627faf94394612641c60f35255c6da0c7f` | `5fc600426233c4b767016cf10596cb9b3189093d19b77dbeb6fa6babe656f2f8` |
| `mozrobot.py` | `464e3d8f1db9f2c0374cbd88a4fda2ae79767aef7389a927c08fee50e3ffce17` | `de7f93ef36f80b47984000fb33d2ac65eda427bdba527dbc7009c270aef9a7d8` |

That original source feeds `resample(T=1)` into PCHIP/SLERP and fails, clears busy by estimated time, and leaves `cancel_actions()` empty. The patch adds a one-row path, tracked background Future/errors, and cancellation of further Python SDK submissions. Future completion means dispatch finished, not that the robot reached its pose. Cancellation cannot withdraw targets already accepted by the SDK/controller.

## Source Package, Backup and Restart

On 2026-11-19 the user manually deployed the patch on the robot and passed strict preflight as `robot-server/0.3.1-rlinf-compat`. Workstation `/home/deng/Documents/codebase/robot_server_code/` retains the original source. The compatible copy is `/home/deng/Documents/codebase/robot_server_code_rlinf_compat_20261119/`; its `rlinf_server_source_compat_20261119.tar.gz` contains two source files, before/after checksum lists and a manifest. This is source, not a Docker image.

To reproduce that deployment, upload the existing package from the workstation:

```bash
scp /home/deng/Documents/codebase/robot_server_code_rlinf_compat_20261119/rlinf_server_source_compat_20261119.tar.gz \
  spirit-ai@172.16.0.30:~/robot_server_code/
```

Verify and deploy on the robot. The original before-check applies only to unpatched files; an already-updated robot should verify after instead of forcing the old check:

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

Each mutation proceeds only after the previous command succeeds. Source changes need a service stop/restart. Fingerprint inspection can run in another robot session while the service stays up:

```bash
docker exec robot_server sha256sum \
  /root/robot_server/server.py /root/robot_server/mozrobot.py
```

If the copy/package is lost, regenerate from supported original source into a new directory. Do not rerun this against an existing output:

```bash
.venv/bin/python -m toolkits.realworld_check.prepare_spiritai_server \
  --source-root /home/deng/Documents/codebase/robot_server_code \
  --output /path/to/new_compat_copy
```

The tool verifies exact source hashes, copies files, applies the archived patch, and checks patched hashes, producing `robot_server/` and `compatibility.json`. The [patch](../pi05-explicit-subtask-ce-lora/artifacts/robot_server_compat.patch) and [version manifest](../pi05-explicit-subtask-ce-lora/artifacts/robot_server_compat_manifest.json) are archived in Git documentation and do not depend on `/tmp`. For packaging, generate `before.sha256` / `after.sha256` from manifest `source_sha256` / `patched_sha256`, then archive only the two sources, checksum lists, and manifest; preserve camera and main.

After listening starts, run strict preflight from the README. It dispatches no actions and changes no following state. `external_following_enabled=true` reports an already-enabled mode; `actions_sent=0` describes only that check. Capability declarations still need on-site confirmation of SDK units, motion direction and controller response through a short test.

## W&B Versus Weight-Loading Failures

The first inference attempt failed in `EmbodiedEvalRunner.__init__ → MetricLogger → wandb.init`, before `runner.init_workers()` and weight loading. As in earlier training, credentials targeted `api.forge.coreweave.com` while the new process defaulted to `api.wandb.ai`.

Inference and RTC do not require W&B. README commands set `runner.logger.logger_backends=[]`, retaining terminal and RTC JSONL. To retain cloud metrics, remove that override and set the matching endpoint in the same launch environment:

```bash
export WANDB_BASE_URL=https://api.forge.coreweave.com
```

A running CLI/service does not automatically set this Python process's endpoint. Keep credentials out of YAML, commands and documentation. Check matching endpoints before investigating authentication further; `Error executing job` alone does not imply corrupt weights.

## Diagnose by Stage

Identify whether startup, inference, RPC or cleanup failed, then use the evidence below. Preserve resolved configuration, trace and observed behavior before changing multiple parameters.

| Symptom | Evidence and response |
|---|---|
| Harbor timeout / image not found | Load the tar and match the image tag; an adjacent tar is not an imported image |
| No listening line yet | SDK/ROS2 or cameras may still be initializing; Docker Up alone is insufficient |
| Immediate disconnect after connect | Read the client error; here strict metadata checks failed after hello, before obs/model execution |
| Missing execution capabilities | Read without `--require-execution`, then inspect actual source; flags or bypasses alone are insufficient |
| Tens of seconds loading without motion | Check hashing/loading first; the profile waits for warmup before dispatch |
| Multiple GPU placements printed | Framework resource enumeration differs from the actual single rollout rank/model process |
| `Evaluating RTC Rollout: Nit` | Requests, not robot ticks; use `episode_len` and trace `step` counts |
| Replan deadline / exhausted chunk | Compare request/inference latency, RPC and action index against remaining horizon; do not repeat the last target indefinitely |
| `reward=0 / success=0` | No automatic folding judge is installed; use observed actions, trace and manual task results |
| `close_error` / following retained | Inspect cancellation ACK, SDK errors and mode response; software cancellation is not an emergency stop |

Ctrl+C requests software stop/cleanup. Hard process/node failures depend on disconnect handling. Before recovery, release old clients, restore the site starting state, and bootstrap again. See [verification](implementation.en.md) for Full Flow hardware results and [parameters](parameters.en.md) for controls.
