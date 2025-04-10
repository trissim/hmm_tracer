
# ------------------------------------------------
# alva_machinery.distribution and .packing 
## are for Visualizing Tcell clonal distribution
The related repository contains code for mathematical visualization of T-cell receptor sequencing data by Power-law, Yule-Simon, and Ewens statistical distributions initially described in the paper: 






# ----------------------------
# alva_machinery.markov
## is for Identifying neurite by RRS method
The related repository contains code for implementing the RRS method initially described in the paper: 

<https://doi.org/10.1038/s41598-019-39962-0>

<https://www.nature.com/articles/s41598-019-39962-0>
```
Random-Reaction-Seed Method for Automated Identification of Neurite Elongation and Branching
by Alvason Li (2019)
```
(is still working on this repository, a new AlvaHmm package will be ready soon...)
## Overview
### tracing neurite in microfuidic device
![](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/figure/AlvaHmm_demo_edge_detection_selected_seeding_selected_seed_window0.jpg)
![](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/figure/AlvaHmm_demo_edge_detection_selected_seeding_connected_way_window3.png)

### Prerequisites
This code is written and tested in Python 3.6.5.
The required Python libaries are:
* NumPy
* SciPy
* Matplotlib