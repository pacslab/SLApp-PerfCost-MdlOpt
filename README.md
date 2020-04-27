# Modeling and Optimization of Performance and Cost of Serverless Applications
Artifact repository for the paper "Modeling and Optimization of Performance and Cost of Serverless Applications." This reproducible repository includes all algorithms, scripts, and experimental results in the paper.
## Overview of the proposed approach
![alt text](docs/PaperIcon.png "Overview of the proposed approach for modeling and optimization of performance and cost of serverless applications. The three boxes from left to right illustrate the input, performance and cost models and optimization algorithms, and corresponding output, respectively. The cloud provider part can be replaced with any FaaS platforms. We use AWS for experimental evaluation in this work.")
* Two analytical models to accurately get the average end-to-end response time and cost of serverless applications.
* Probability refined critical path algorithm to answer the follwing two optimization problems for non-DAG serverless workflows. 
    * Best performance (end-to-end response time) under a budget constraint
    * Best cost under a performance constraint
* Models and algorithms verified by experiments on AWS.
## Artifacts
* [Performance and Cost Models](./source/ServerlessAppPerfCostMdlOpt/ServerlessAppWorkflow)
* [Probability Refined Critical Path Algorithm](./source/ServerlessAppPerfCostMdlOpt/PerfOpt)
* [Servleress Application Generator for Testing and Experimental Evaluations](./source/ServerlessAppPerfCostMdlOpt/AppGenerator)
* [Scripts and Resulsts of Experimental Evaluations](./evaluations)
    * [Experimental Evaluations of Performance and Cost Models](./evaluations/model)
        * [Result Analysis](./evaluations/model/analysis/ResultAnalysis.ipynb)
        * [App8](./evaluations/model/App8/App8.ipynb)
        * [App10](./evaluations/model/App10/App10.ipynb)
        * [App12](./evaluations/model/App12/App12.ipynb)
        * [App14](./evaluations/model/App14/App14.ipynb)
        * [App16](./evaluations/model/App16/App16.ipynb)
    * [Experimental Evaluations of the Probability Refined Critical Path Algorithm](./evaluations/alg)
        * [Result Analysis](./evaluations/alg/analysis/ResultAnalysis.ipynb)
        * [App1](./evaluations/alg/App1/App1.ipynb)
        * [App2](./evaluations/alg/App2/App2.ipynb)
        * [App3](./evaluations/alg/App3/App3.ipynb)
        * [App4](./evaluations/alg/App4/App4.ipynb)
        * [App5](./evaluations/alg/App5/App5.ipynb)
        * [App6](./evaluations/alg/App6/App6.ipynb)
* [Worst Case Scenario Analysis](./evaluations/model/WorstCaseScenario/WorstCaseScenario.ipynb)
* [AWS Step Functions Interaction Delay Analysis](./evaluations/model/SFNDelay/StepFunctionsDelay.ipynb)

## Requirements
* Python>=3.7
* pip>=19.1.1

Required Python packages are listed in requirements.txt

Installation:
```
pip install -r requirements.txt
```

## Usage
Please follow this [documentation](./docs/usage.ipynb).

## Figures
Hover to see the caption. Self-loop edges might be overlapped.
### Evaluation Results

<img src="./evaluations/model/analysis/Analytical_Model_Accuracy_RT.png" style="float: left" width="500" title="Experimental evaluation result of the performance model. As the number of functions in the application increases from 8 to 16, the workflow become more complex in terms of structures. The average accuracy is 98.75%.">
<img src="./evaluations/model/analysis/Analytical_Model_Accuracy_Cost.png" style="float" width="500" title="Experimental evaluation result of the cost model. The average accuracy is 99.97%">

<img src="./evaluations/alg/App6/App6_Optimization_Curve_BPBC.png" style="float: left" width="500" title="The result of the PRCP algorithm solving the BPBC problem. BCR threshold is 0.2. The budget constraints are 100 equidistant values between 58.86 (minimum cost) and 163.90 (maximum cost). The average accuracy of the best answer is 97.40%.">
<img src="./evaluations/alg/App6/App6_Optimization_Curve_BCPC.png" style="float" width="500" title="The result of the PRCP algorithm solving the BCPC problem. BCR threshold is 0.2. The performance constraints are 100 equidistant values between 2748.24 ms (minimum ERT) and 25433.08 ms (maximum ERT). The average accuracy of the best answer is 99.63%.">


![alt text](./evaluations/model/WorstCaseScenario/App27_G.png "The worst case scenario with 27 functions and 406 edges. The performance and cost models can get results for the worst case with 27 functions and 406 edges in 1 second on a laptop with a 2.70GHz Intel Core i7-3740QM processor and 16 GB of memory.")

<img src="./evaluations/alg/analysis/Iterations.png" style="float" width="500" title="the number of iterations grows linearly with the increasing number of functions as well as structures in the application.">


### Workflow of Apps
![alt text](./evaluations/model/App16/App16_G.png "App16")
![alt text](./evaluations/model/App14/App14_G.png "App14")
![alt text](./evaluations/model/App12/App12_G.png "App12")
![alt text](./evaluations/model/App10/App10_G.png "App10")
![alt text](./evaluations/model/App8/App8_G.png "App8")
![alt text](./evaluations/alg/App6/App6_G.png "App6")
![alt text](./evaluations/alg/App5/App5_G.png "App5")
![alt text](./evaluations/alg/App4/App4_G.png "App4")
![alt text](./evaluations/alg/App3/App3_G.png "App3")
![alt text](./evaluations/alg/App2/App2_G.png "App2")
![alt text](./evaluations/alg/App1/App1_G.png "App1")




## Abstract of the Paper
Function-as-a-Service (FaaS) and Serverless applications have proliferated significantly in recent years because of their high scalability, ease of resource management, and pay-as-you-go pricing model. However, cloud users are facing practical problems when they shift their applications to serverless pattern, which are the lack of analytical performance and billing model and trade-off between limited budget and Service Level Agreement (SLA) guaranteed performance of serverless applications. In this paper, we fill this gap by proposing and answering two research questions regarding the prediction and optimization of performance and cost of a serverless applications. We propose the definition of the serverless workflow, and implement analytical models to predict the average end-to-end response time and cost of the application, giving practical solutions to the current unpredictable performance and cost problems for serverless applications. We propose a heuristic algorithm named Probability Refined Critical Path Greedy algorithm (PRCP) with four greedy strategies to answer two types of optimization questions regarding performance and cost. We experiment proposed models with five serverless applications deployed on AWS. Our results show that the performance and cost models can predict the performance and cost of serverless applications with more than 97.5\% of accuracy. PRCP can give the optimized memory configurations of functions in serverless applications with over 97\% of accuracy.
