# End-to-end antigenic variant generation for H1N1 influenza HA protein using sequence to sequence models
## Abstract
The growing risk of new variants of the influenza A virus is the most significant to public health. The risk imposed from new variants may have been lethal, as witnessed in the year 2009. Even though the improvement in predicting antigenicity of influenza viruses has rapidly progressed, few studies employed deep learning methodologies. The most recent literature mostly relied on classification techniques, while a model that generates the HA protein of the antigenic variant is not developed. However, the antigenic pair of influenza virus A can be determined in a laboratory setup, the process needs a tremendous amount of time and labor. Antigenic shift and drift which are caused by changes in surface protein favored the influenza A virus in evading immunity. The high frequency of the minor changes in the surface protein poses a challenge to identifying the antigenic variant of an emerging virus. These changes slow down vaccine selection and the manufacturing process. In this vein, the proposed model could help save the time and efforts exerted to identify the antigenic pair of the influenza virus. The proposed model utilized an end-to-end learning methodology relying on deep sequence-to-sequence architecture to generate the antigenic variant of a given influenza A virus using surface protein. Employing the BLEU score to evaluate the generated HA protein of the antigenic variant of influenza virus A against the actual variant, the proposed model achieved a mean accuracy of 97.57%.

## Sequence to Sequence Model

![Sequence to Sequence Model]( /images/seq_to_seq.PNG "Sequence to Sequence Model") 


## Attention Layer

![Attention.PNG](/images/Attention.PNG "Attention Layer")

## Experimental Models Architectures

![Experimental Models Architectures](/images/Architectures.PNG "Experimental Models Architectures")

## Hyper-Parameters

![Hyper-Parameters](/images/hyper.PNG "Hyper-Parameters")


## Evaluation Metrics
 The generated sequences acquired during the test phase are evaluated against the original sequences. Since the approach taken in this paper is similar to approaches used in NLP, it was convenient to use accuracy metrics used in NLP as well. Hence, the utilization of the BLEU score which is often used in NLP problems. BLEU score consists of two main terms, particularly precision score and a brevity penalty. The precision score, calculated for each n-gram length by summing over the match for a generated sequence _S_ against the original sequence _O_ from the test set.
 
![Hyper-Parameters](/images/BLEU.PNG "Hyper-Parameters")

## Results

During the training stage, architectures utilizing bi-directional recurrent layers—both LSTM and GRU—performed better than other architectures in terms of both training and validation accuracy. Unexpectedly, the addition of an attention layer to the encoder-decoder architecture did not improve training or validation accuracy. Moreover, in the case of using bi-directional LSTM recurrent layer in encoder and decoder with additional attention layer, both the training and validation accuracy were lower than the chosen baseline architectures.

![Training](/images/Train1.PNG "Training")


 Improvement are shown in training accuracy and loss compared to the validation over epochs for the architecture utilizing bi-directional GRU as a recurrent layer. However, the improvement in accuracy was very small after approximately 60 epochs. It is notable that, the loss and accuracy of both training and validation are almost identical, with a very small generalization gap. Moreover, both loss and accuracy reach a point of stability at the end of training epochs. This typically indicates that the model is a good fit.

![Training Versus Validation](/images/Train_valid.PNG "Training Versus Validation")


Surprisingly, the best-performing architecture was the one based on bi-directional GRU layers, despite the expectations that the introduction of attention would improve accuracy. Nevertheless, the bi-directional LSTM with attention and deep GRU performed less than baseline architectures.

![Mean Accuracy](/images/Mean Accuracy.PNG "Mean Accuracy")

Additionally, the bi-directional GRU architecture achieved a higher mean accuracy. However, the gap between this architecture and attention-based architectures is fairly high, which indicates that attention might not be helpful in this case.
The leading architecture (Deep_BI_GRU), when evaluated with a 1-gram BLEU score shows a maximum accuracy of 100% and a median accuracy of 98.5% approximately where the first quartile is above 96%. In other words, generated sequences accuracy distribution is right-skewed with a minimum accuracy of 92%, which indicates a high quality in generated sequences when compared to original sequences. On the other hand, the LSTM architecture shows more diversity in results with a lower mean accuracy of 96% approximately and a minimum accuracy below 92%. Finally, the bi-directional LSTM with attention model shows even higher variations in generated sequences quality where the minimum falls below 88%. clearly proposed architectures have consistent accuracy distributions over 4-gram BLEU scores where the bi-directional GRU architecture is explicitly performing better than other architectures.

![Accuracy Distribution](/images/Accuracy Distribution.PNG "Accuracy Distribution")


## Conclusion

In this work, a single-step methodology was adopted that does not require feature engineering, and it is not dependent on virus discovery chronological order in training or prediction. Hence, this methodology requires less computational effort. Furthermore, integrating this methodology into existing influenza virus databases will alleviate a colossal part of antigenic pair identification and vaccine design.

Finally, the metric utilized in this work namely BLEU is generally accepted in NLP problems. Since there is no conventional computational method for measuring the quality of generated sequences, the BLEU score is seen as an effective metric in measuring the quality of generated sequences.