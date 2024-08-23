# infra_utils
tool to test our infra

Most code can be run as single script using [uv](https://github.com/astral-sh/uv)

## all reduce multi node

First get the ip of the first node:

```bash
ip a
```

then on each node:

install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

export env vars
```bash
export RDZV_ENDPOINT=<ip_of_node_0>:1234
export MY_RANK=<rank_of_node>
```

then to run the scrit

```bash
uv run --with torch --with numpy --with setuptools torchrun --nproc_per_node=8 --node-rank $MY_RANK --rdzv_endpoint=$RDZV_ENDPOINT --nnodes=<total_number_of_nodes> all_reduce_test.py --n_iters 5
```