# VGGPyTorch
A comprehensive tutorial on how to implement and train a VGG model using PyTorch

## Demo notebooks
1. Before training the VGG model, please first use the [VGG_Preprocessing.ipynb](./VGG_Preprocessing.ipynb) notebook to calculate the channel average value, and channel PCA eigenvectors and eigenvalues on the training dataset, which is required by the data processing and augmentation during training. Copy the results to the traininng notebooks ([VGG_Train.ipynb](./VGG_Train.ipynb) and [VGG_Train_w_PreTrainModel.ipynb](./VGG_Train_w_PreTrainModel.ipynb)).
2. To train shallow models, please use the [VGG_Train.ipynb](./VGG_Train.ipynb) notebook.
3. To train deep models, which requires transfering learned paremeters from pretrained models, please start with  the [VGG_Train.ipynb](./VGG_Train.ipynb) notebook and then use[VGG_Train_w_PreTrainModel.ipynb](./VGG_Train_w_PreTrainModel.ipynb) to transfer learned parameters from pretrained models and perform training. 

## Tutorial
A step by step tutorial on how to build and train VGG using PyTorch can be found in my [blog post](https://jianzhongdev.github.io/VisionTechInsights/posts/implement_train_vgg_pytorch/) (URL: https://jianzhongdev.github.io/VisionTechInsights/posts/implement_train_vgg_pytorch/) 

## Dependency
This repo has been implemented and tested on the following dependencies:
- Python 3.10.13
- matplotlib 3.8.2
- numpy 1.26.2
- torch 2.1.1+cu118
- torchvision 0.16.1+cu118
- notebook 7.0.6

## Computer requirement
This repo has been tested on a laptop computer with the following specs:
- CPU: Intel(R) Core(TM) i7-9750H CPU
- Memory: 32GB 
- GPU: NVIDIA GeForce RTX 2060

## License

[GPL-3.0 license](./LICENSE)

## Reference

[1] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for Large-Scale image recognition. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1409.1556

[2] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. Neural Information Processing Systems, 25, 1097–1105. https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

[3] Krizhevsky, A., Nair, V. and Hinton, G. (2014) The CIFAR-10 Dataset. https://www.cs.toronto.edu/~kriz/cifar.html

## Citation
If you found this repo helpful, please cite it as:

> Zhong, Jian (May 2024). Building and Training VGG with PyTorch: A Step-by-Step Guide. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/implement_train_vgg_pytorch/.

Or

```html
@article{zhong2024buildtrainVGGPyTorch,
  title   = "Building and Training VGG with PyTorch: A Step-by-Step Guide",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "May",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/implement_train_vgg_pytorch/"
}
```