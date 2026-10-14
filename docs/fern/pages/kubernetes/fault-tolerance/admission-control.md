---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Admission Control
subtitle: Configure and verify the backend admission gate in a Kubernetes deployment.
---

Configure NVIDIA Dynamo backend admission control on each worker component in your
DynamoGraphDeployment (DGD). The gate bounds the requests inside the engine and, among them, the
requests still waiting for their first generated token, or a prefill for its completion, and queues
excess work outside the engine. Each limit applies per data-parallel (DP) rank.

For the design, rank handling, cancellation, metrics, and sizing guidance, see
[Admission Control Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md).

## Configure the Worker

Start with a working [DynamoGraphDeployment](../../reference/kubernetes-api/dynamo-graph-deployment.mdx).
Configure the backend admission gate through these three environment variables on the worker container:

| Environment variable | Purpose, per DP rank | Default |
|---|---|---|
| `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT` | Limits admitted requests that have not finished. Each request holds this capacity until its response stream ends. | The engine's reported `max_num_seqs`; `10000` until reported. |
| `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT` | Limits admitted requests still waiting for their engine's output. A decode, aggregated, or encode request holds this capacity until its first generated token, a prefill request until its completion response, and any request at most until its response stream ends. | The engine's reported `max_num_seqs`; `10000` until reported. |
| `DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT` | Limits requests waiting outside the engine in each rank's gate queue. Set `0` to disable queueing. | Twice the engine's reported `max_num_seqs`; `20000` until reported. |

A request enters the engine only when both engine limits have room, for its own rank and across the
worker process. A worker serving `N` ranks allows `N` times each limit in total. A request the gate
refuses for capacity reaches the client as HTTP 529 under the default frontend configuration.

The defaults are starting settings derived from the engine's capacity; tune them for your workload. The
following `nvidia.com/v1beta1` fragment sets an engine request limit of `256`, an engine request
waiting limit of `32`, and a queue limit of `512`, each per rank. Omit an entry to keep its default.

Merge these entries into the existing worker container's `env` list, retaining its image,
arguments, resources, and other environment entries.

```yaml
spec:
  components:
    - name: worker
      type: worker
      podTemplate:
        spec:
          containers:
            - name: main
              env:
                - name: DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT
                  value: "256"
                - name: DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT
                  value: "32"
                - name: DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT
                  value: "512"
```

For a disaggregated deployment, apply the settings to both the prefill and decode worker components,
choosing limits for each stage. Each worker process has an independent gate. Set these variables on
the workers rather than on the frontend.

The worker's `--engine-request-limit` argument and the `DYN_ENGINE_REQUEST_LIMIT`,
`DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT`, and `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` variables configure the
same three limits. `--engine-request-limit` takes precedence over both engine request limit variables,
and each variable above takes precedence over its alias. `DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT` is the
former name of the waiting limit; rename it in existing deployments. See
[Configuration](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#configuration)
for accepted values and precedence.

> [!WARNING]
> A TCP request keeps its worker-pool permit while it waits at the gate. For every queue place to be
> reachable, set `DYN_TCP_WORKER_POOL_SIZE` to at least `N * (engine request limit + queue limit)` for a
> worker serving `N` ranks. NATS requests do not use this pool.

## Verify the Configuration

Check the worker's `Backend admission gate created` log entry for the effective limits, their sources,
and the rank count. When the engine reports its capacity, the worker logs the updated limits, and it
warns about any limit more than 10 times the reported `max_num_seqs`.

Use the `dynamo_backend_admission_*` metrics to observe engine and queue occupancy against the
effective limits, and rejections and cancellations under load. A limit lowered by a capacity report
evicts nothing, so occupancy can exceed it until those requests finish. Every series carries a
`dp_rank` label, `unspecified` for requests without a usable rank; sum over `dp_rank` for worker totals.
See [Metrics](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#metrics)
for the names and their meanings.
