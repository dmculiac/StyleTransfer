# Neural Style Transfer
<p align = 'center'>
  <img src = "https://images.squarespace-cdn.com/content/54856bade4b0c4cdfb17e3c0/1478946037611-W80NSF6WA0DRXE31SBCS/?content-type=image%2Fjpeg", width='300' >
  <img src = "https://i.ytimg.com/vi/mGE64KyNtu4/hqdefault.jpg", width='300'>
  </p>

#### Why this particular topic to begin with
In a regression or classification problem, the **input is fixed** and we optimize an objective function with regard to parameters, i.e. we tweak the parameters until the objective function is optimized. In a neural style transfer, the **optimization is performed on the input data** while the parameters are preset (fixed, pre-trained). So this represents a drastically different angle for optimization than we usually take. But there's more to it. This offers an opportunity to tackle the semantics of style in a quantitative way. 

#### Intuition for NST
We'll generate an image that is a blend between the content of a input image and the style of another input image. We'll acomplish this by balancing the content loss and the style loss of the blended image simultaneously.

Convolutional Neural Networks (CNNs) are very good at extracting features from an image: horizontal, vertical, diagonal, triangles, circles, squares etc. So when we're passing as input the content picture, the CNN extracts just the essence of the input. The rest is filled with the style coming form the second input.

#### Style
But what is style? Abstract concept. Algorithm-wise, the style is the result of performing convolutions on the input image that serves as the style template, computing its Gram matrix and using the ensuing result in the style square loss calculation. Sounds like a mouthful. Why does this even work? 

#### Autocorrelation 
The functional form of the Gram matrix looks very close to the autocorrelation matrix. Autocorrelation is the correlation of the data with itself. It appears often in the context of time series and signal processing. It has multiple applications - discovering the signal in the noise and uncovering seasonal patterns: patterns that repeat with time. So what we're essentially doing is extracting the patterns of the input image designated as the style input. In this particular context, style is a form of repeating pattern. So there we have it, the intuition behind the abstraction.
