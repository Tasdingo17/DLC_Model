# DLC model

The tool for generating packet losses co-dependently. 

Current version is just python packege for research of tool characteristics. Linux kernel module (netem-like) development is future work.

## Prerequisites

python >= 3.10

git

## Start

Clone repository

```git clone https://github.com/Tasdingo17/DLC_Model.git```

Install python3 packages

```pip3 install -r requirements.txt```

## Usage example

Usage example is the `example.py` script.

At the start of the file there are constants for model's parameters and the number of samples to generate.
The script will generate samples, print some statistics and draw plots (like `gen_combined_example.svg`).

Feel free to modify the script :).

## Experiments

There are two folders with experimental research:

- `accuracy_experiments`. Experiments to research models accuracy within generating target parameters. Some results are also inside.

- `real_comparison`. Comparison with real traffic (iperf3 flow from Moscow to Sasovo by VPN-channel). Traffic was collected on both endpoints, extracted by tshark to `msk2sas.csv` and `sas2msk.csv`, then parsed and anylized by `pcap_parser.py`. 
Finally, statistics that correspond to DLC model were calculated, provided to `example.py`-like file and plots were drawn.
