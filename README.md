# Deep Learning Project 1 - Symplectic Gradient Adjustment

This is one of the Artificial Neural Network course projects. In the project, a new gradient descent method and algorithm called Symplectic Gradient Adjustment (SGA) is implemented. (For the privacy sake I delete all my real name here.)

## Project Abstract

The state-of-the-art deep learning model is guaranteed by the gradient descent method to converge to a local minimum. 
However, this guarantee fails in settings with multiple interactive loss functions, especially generative adversarial networks (GAN). Little research deepens into the mechanism of how interactive loss functions affect the performance of the gradient descent method, and only a few adjustments to the current gradient descent method are proved effective in GAN. 
The authors propose a new adjustment to the current gradient descent method called Symplectic Gradient Adjustment (SGA) to find the stable fixed point in n-player games e.g. GAN. The key to SGA is to decompose the second-order dynamics, which is also known as Hessian, into two components. Both are easy to solve and can converge to Nash equilibrium by gradient descent, and thus they build up mechanics to solve general n-players differentiable games. 
I conduct a series of experiments to evaluate its performance, including four major sections. Firstly, I make a comparison to classic gradient-descent-based optimizers in a Gaussian Mixture Model (GMM). Secondly, a state-of-the-art optimizer, optimistic mirror decent (OMD), is used to analyze the threshold of learning rate. Thirdly, we introduce a high dimensional GMM to identify the universality of the SGA. Finally, a Pytorch implementation of SGA is proposed. The experiments' result shows the effectiveness of SGA in GMM.


## File Structure

- `project.ipynb` the jupyter notebook file of the SGA implementation and benchmark (in Tensorflow)
- `project.pdf` the pdf version of project.ipynb
- `report.pdf` the technical report of the project
- `summary.docx` the summary of the SGA paper
- `midterm_report/` the mid-term report of the whole project