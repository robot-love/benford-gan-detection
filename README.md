# Detecting Deepfakes using Benford's Law

This is an implementation of the paper titled "On the use of Benford's law to detect GAN-generated images" ([PDF](https://arxiv.org/abs/2004.07682)). Training and testing is done using the DoGANs dataset ([link](https://grip-unina.github.io/DoGANs/))

## A Quick Primer

Benford's Law is a power-law probability distribution based on the distribution of leading digits in natural number sets. It's one of the tools commonly used in finance and accounting to detect fraud, such as falsification of records. The distribution for base 10 is given by:

$$
p(d) = \log_{10} \bigg(1+\frac{1}{d} \bigg)
$$

where $d$ is the first digit. 


## Usage
