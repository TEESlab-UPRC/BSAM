![BSAM-02](https://user-images.githubusercontent.com/118806905/203372814-297aeb0f-8c47-425b-85f0-fa3965e2e8c2.jpg)

## Contents
- [Contents](#contents)
- [About](#about)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Citing BSAM](#citing-bsam)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About
BSAM is an agent-based electricity wholesale market simulation model which simulates the operations within a power pool central dispatch Day Ahead Market. The model simulates electricity generators as entities who progressively learn to bid their capacities in a day-ahead competitive wholesale market, with ultimate goal the maximization of their profits. In parallel, a unit commitment and economic dispatch algorithm calculates the cost-optimal power mix to satisfy demand, the quantities injected by each generation unit, the market clearing price, as well as, derived outputs such as CO2 emissions and profits of each generator. The model can support cost-benefit analysis of future policy and/or technology deployment scenarios.

It consists of three modules: (i) a wholesale electricity market module which models the market operations (i.e. setting of the power market needs, enforcement of market price caps, determination of water prices, etc.), (ii) an agent-based module which models the bidding behaviour of electricity generating resources (as self-learning profit-maximizing agents) participating in a Day-Ahead competitive wholesale electricity market, and (iii) a unit commitment module that calculates the cost-optimal commitment and dispatch of generating resources. BSAM is written in Python and combines a reinforcement learning approach to model agent-based decision-making, with heuristic algorithms enabling fast and accurate simulations of the Unit Commitment and Economic Dispatch problems. It is maintained by the [Techno-Economics of Energy Systems laboratory (TEESlab)](https://teeslab.unipi.gr) at the University of Piraeus and is freely available on GitHub. 

## Quick start
* Install Python 3.9
* Download BSAM from Github and save it in a folder of your preference
* Using a terminal (command line) navigate to the folder where BSAM is saved 
* Type pip install -r requirements.txt
* Type python main.py to run the preconfigured example

## Documentation
Read the full [documentation](http://teeslab.unipi.gr/bsam-documentation_v1-0/)

## Citing BSAM
In academic literature please cite BSAM as: 
>[![article DOI](https://img.shields.io/badge/article-10.1016/j.egyr.2021.07.052-blue)](https://doi.org/10.1016/j.egyr.2021.07.052) Kontochristopoulos, Y., Michas, S., Kleanthis, N., & Flamos, A. (2021). Investigating the market effects of increased RES penetration with BSAM: A wholesale electricity market simulator. *Energy Reports*, *7*, 4905-4929. [https://doi.org/10.1016/j.egyr.2021.07.052](https://doi.org/10.1016/j.egyr.2021.07.052)


## License
The **BSAM source code**, consisting of the *.py* files, is licensed under the GNU Affero General Public License:

    Copyright (C) 2022 Technoeconomics of Energy Systems laboratory - University of Piraeus Research Center (TEESlab-UPRC)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
The input data contained in the **Data** folder are collected through publicly available sources, or are modified/simplified versions of private data. BSAM license does not apply to input data.

## Acknowledgements
The development of BSAM has been partially funded by the following sources:
* The EC funded Horizon 2020 Framework Programme for Research and Innovation (EU H2020) Project titled "Sustainable energy transitions laboratory" (SENTINEL) with grant agreement No. 837089
* The EC funded Horizon 2020 Framework Programme for Research and Innovation (EU H2020) Project titled "Transition pathways and risk analysis for climate change policies" (TRANSrisk) with grant agreement No. 642260
