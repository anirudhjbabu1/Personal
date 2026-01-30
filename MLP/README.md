**Multi-Layer Perceptron (MLP)**

script includes He Initialization, ReLU/Sigmoid activations, Cross-Entropy Loss, and a Manual Training Loop.


***OUTPUT***
```
ai@AI:~$ python3 mlp.py 
Epoch 0 | Loss: 0.6539
Epoch 100 | Loss: 0.1613
Epoch 200 | Loss: 0.0977
Epoch 300 | Loss: 0.0755
Epoch 400 | Loss: 0.0627
ai@AI:~$ python3 mlp.py 
Epoch 0 | Loss: 0.7136
Epoch 100 | Loss: 0.1572
Epoch 200 | Loss: 0.0841
Epoch 300 | Loss: 0.0601
Epoch 400 | Loss: 0.0466
ai@AI:~$ python3 mlp.py 
Epoch 0 | Loss: 0.6838
Epoch 100 | Loss: 0.1804
Epoch 200 | Loss: 0.1037
Epoch 300 | Loss: 0.0798
Epoch 400 | Loss: 0.0670
ai@AI:~$ python3 mlp.py 
Epoch 0 | Loss: 0.7332
Epoch 100 | Loss: 0.1591
Epoch 200 | Loss: 0.0905
Epoch 300 | Loss: 0.0682
Epoch 400 | Loss: 0.0571
ai@AI:~$ python3 mlp.py 
Epoch 0 | Loss: 0.7272
Epoch 100 | Loss: 0.1790
Epoch 200 | Loss: 0.1038
Epoch 300 | Loss: 0.0759
Epoch 400 | Loss: 0.0609
```
Initial High Loss: At Epoch 0, the loss is high (around 0.7). This is because the model's weights are random—it is essentially "guessing" blindly.

The "Learning" Phase: As the epochs progress, you’ll see the number decrease. This proves the Backpropagation is working; the model is successfully adjusting its weights to reduce the error.

Convergence: By the end, the loss should be very low (near 0.0), meaning the model has "solved" the mathematical relationship in your random data.
