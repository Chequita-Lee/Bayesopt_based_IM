# Database and Python Codes for optimal IM identification based on Bayesian optimization

All the data and code you need to run the program are stored in the zip **"code share"**.<br>
After downloading the zip "code share", unzip the file and **run the ".py" in the zip file** instead of the '.py' out of the zip file.<br>
  
Pure Python implementation of bayesian global optimization with gaussian processes.<br>
* PyPI (pip):  
```
$ pip install bayesian-optimization
```
  
* The folder 'PULSE': The pulse-like ground motions selected in this study are saved in this folder. <br>
  
 * The folder **'Set_1a_processed'**, **'Set_1b_processed'**, **'Set_2_processed'**: <br>
    *   The ground motions in Sets 1a, 1b, and 2 are selected by Baker et al. as a part of the PEER Transportation Research Program. <br>
    *   **Set 1a** corresponds to ground motions with M_w=7 R_rup=10 km under rock site; <br>
    *   **Set 1b** corresponds to ground motions with M_w=6 R_rup=25 km under soil site; <br>
    *   **Set 2** correspond to ground motions M_w=7 R_rup=10 km under soil site.<br>
  
* The folder **'Set_4_processed'**: the Sa, Sv, and Sd correspond to the 121 pulse-like ground motions in the folder "Pulse".<br>

* **Xtrain_cable_trans.xlsx**: The EDPs obtained by nonlinear time history analyses of a cable-stayed bridge.<br>


* **ASI_trans.py**: Example of the identification of optimal Sa and ASI based on Bayesian optimization<br>

* **VSI_trans.py**: Example of the identification of optimal Sv and VSI based on Bayesian optimization<br>

* **IH_trans.py**: Example of the identification of optimal Sd and IH based on Bayesian optimization<br>
