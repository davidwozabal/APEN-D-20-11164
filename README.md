# How much smart charging is smart?

This python package includes a toolset develop and utilized to assess the impact
of a controlled plug-in electric vehicle (PEV) charging and variables renewable 
energy source (VRES) curtailment policy on distribution system cost.

It has been developped for research purposes of a research project titles "How 
much smart charging is smart?" by Christoph Heilmann and David Wozabal. The 
following abstract provides and overview on the content and findings:

The threat of climate change motivates the expansion of variable renewable energy
sources (VRES) as well as the transition to plug-in electric vehicles (PEV). Both
technologies present unique challenges to electricity systems, potentially leading
to high grid expansion cost. In this paper, we present a detailed assessment on 
how smart-charging of PEVs can mitigate these problems and reduce cost on the 
distribution grid level. To this end, we propose a heuristic policy that dynamically
decides on charging of PEVs, curtailment of VRES, and subsequently on infrastructure
investments trading off fixed investments with variable operational costs. The main 
inputs are modeled as stochastic and the proposed policy is non-anticipative. We 
conduct a comprehensive case study for Germany in the year 2035 using detailed 
descriptions of existing distribution grids, realistic driving patterns, and 
real-world VRES feed-in data. Potential savings of EUR 6.2 billion in investment 
costs lead to a reduction in total distributio grid cost of around 19%. This result 
is achieved by upgrading 7 million (21%) PEV chargers all over Germany. Upgrading 
100% of chargers to smart-chargers as is proposed in the extant literature is clearly
sub-optimal as it leads to significantly higher total cost. A closer investigation 
for single grids reveals that savings as well as the optimal share of smart chargers 
varies widely between grids. In particular, the potential of smart charging is much 
greater in rural areas than in urban centers. Furthermore, the results suggest that 
curtailment of VRES production is economically only in rare circumstances.


# Usage and references example

We use the Open Electricity Grid Optimization (open_eGo - https://github.com/openego) 
toolbox to generate synthetic models of German grids as well as load and demand curves 
and perform the distribution grid expansion planning. We use the electricity Distribution 
Grid optimization (eDisGo - https://github.com/openego/eDisGo) to calculate the 
distribution grid expansion needed to avoid overloading and voltage issues in the system.

We optimize the parameters of our smart charging strategy using the Python implementation 
of the covariance matrix adaptation evolution strategy (CMAES - 
https://github.com/CMA-ES/pycma).

Our modeling of random PEV driving behaviour, specifically trip duration, trip times, and 
driving distance are based on real-world driving data from the German Mobility Panel (GMP) 
which supplies trip data from a survey in 2015/2016 and 2016/2017 [1]. As this data is 
not open source, we can not provide it as part of this package.

The package is run from "Run_files/run_file.py". 

[1] German  Federal  Ministry  of  Transport  and  Digital  Infrastructure,  German  
mobility  panel(Deutsches Mobilit ¨atspanel):  Time series 2015/2016 and 2016/17 (2017). 
URLhttps://mobilitaetspanel.ifv.kit.edu/34


# Meta

Christoph Heilmann – Christoph.Heilmann@tum.de
David Wozabal - David.Wozabal@tum.de

LICENSE
-------

Copyright (C) 2020 Christoph Heilmann

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program. If not, see https://www.gnu.org/licenses/.

