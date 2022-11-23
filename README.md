![BSAM-02](https://user-images.githubusercontent.com/118806905/203372814-297aeb0f-8c47-425b-85f0-fa3965e2e8c2.jpg)

## Contents
- [Contents](#contents)
- [About](#about)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Citing BSAM](#citing-bsam)
- [License](#license)

## About
BSAM is an agent-based electricity wholesale market simulation model which simulates the operations within a power pool central dispatch Day Ahead Market. The model simulates electricity generators as entities who progressively learn to bid their capacities in a day-ahead competitive wholesale market, with ultimate goal the maximization of their profits. In parallel, a unit commitment and economic dispatch algorithm calculates the cost-optimal power mix to satisfy demand, the quantities injected by each generation unit, the market clearing price, as well as, derived outputs such as CO2 emissions and profits of each generator. The model can support cost-benefit analysis of future policy and/or technology deployment scenarios.

It consists of three modules: (i) a wholesale electricity market module which models the market operations (i.e. setting of the power market needs, enforcement of market price caps, determination of water prices, etc.), (ii) an agent-based module which models the bidding behaviour of electricity generating resources (as self-learning profit-maximizing agents) participating in a Day-Ahead competitive wholesale electricity market, and (iii) a unit commitment module that calculates the cost-optimal commitment and dispatch of generating resources. BSAM is written in Python and combines a reinforcement learning approach to model agent-based decision-making, with heuristic algorithms enabling fast and accurate simulations of the Unit Commitment and Economic Dispatch problems. It is maintained by the [Techno-Economics of Energy Systems laboratory (TEESlab)](https://teeslab.unipi.gr) at the University of Piraeus and is freely available on GitHub. 

## License
