# Response of Selective Attention in Middle Temporal Area

The primary visual cortex processes a large amount of visual information, however, due to its large receptive fields, when multiple stimuli fall within one receptive field, there are computational problems. To solve this problem, the visual system uses selective attention, which allocates resources to a specific spatial location, to attend to one of the stimuli in the receptive field. During this process, the center and width of the attending receptive field change. The model presented in the paper, which is extended and altered from Bobier et al., simulates the selective attention between the primary visual cortex, V1, and middle temporal (MT) area. The responses of the MT columns, which encode the target stimulus, are compared to the results of an experiment conducted by Womelsdorf et al. on the receptive field shift and shrinkage in macaque MT area from selective attention. Based on the results, the responses in the MT area are similar to the Gaussian shaped receptive fields found in the experiment. As well, the responses of the MT columns are also measured for accuracy of representing the target visual stimulus and is found to represent the stimulus with a root mean squared error around 0.17 to 0.18. The paper also explores varying model parameters, such as the membrane time constant and maximum firing rates, and how those affect the measurement. This model is a start to modeling the responses of selective attention, however there are still improvements that can be made to better compare with the experiment, produce more accurate responses and incorporate more biologically plausible features. 

Check out the paper [here](selective-attention.pdf).

## To run the simulation
- Install nengo 
```python
pip install nengo nengo-gui 
```
- Then run 
```python
nengo selective-attention.py
```