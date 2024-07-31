# if your node follows some pattern, rather than having a specific name,
# e.g. HPC nodes, then enter this pattern here.
# You can then override node_name to point to the appropriate file in env/nodes.

if [[ $node_name == *"glogin"* ]] || [[ $node_name == *"ggpu"* ]]; then
	node_name="gwdg_cuda"
fi

if [[ $node_name == "bgi"* ]]; then
	node_name="zib_intel"
fi
