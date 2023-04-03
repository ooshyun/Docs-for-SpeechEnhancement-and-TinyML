# Papers for Speech Enhancement

@updated date 2022.12.31

## 1. Research
### 1.1 Keywords
- Search
	```
	SE
	Speech Enhancement
	Enhancement of Fullband Speech
	Speech Dereverberation
	Speech Denoising 
	Complex spectrogram enhancement
	Noise Suppression
	Noise Power Spectral Density Estimation
	Speech Speration
	Speech Recognition
	Survey
	```

- Method
	```
	ML Method
		Magnitude spectrum
			spectral masking
			spectral mapping
		Complex domain
		Time domian
		Multi-stage
		Feature augmentation
		Audio-Visual SE
		Netword design
			Filter design
			Fusion technique
			Attention
			U-Net
			GAN
			Auto-Encoder
			Hybrid SE
			SepFormer(Transformer)
		Phase reconstruction
		Learning strategy
			Loss function
			Multi-task learning
			Curriculum learning
			Transfer learning
		Model Compression
		
	Classical
		Wiener filter
		Kalman filter
		minimum variance distortionless response (MVDR) beamformer
	```
### 1.2 Research
#### 2.1 Survey
- H. Purwins, B. Li, T. Virtanen, J. Schlu ̈ter, S. Chang, and T. Sainath, “Deep learning for audio signal processing,” J. Selected Topics Sig. Proc., vol. 13, no. 2, pp. 206–219, 2019.
	- Method
		1. Audio analysis and synthesis
			- Sequence classification
			- Multi-label sequence classification
			- Sequence regression
			- Sequence lableing, ex. Chord annotation, Vocal Activity Detection
			- Sequence transduction
			- Audio synthesis
		2. Representation, Audio Feature
			- Mel Frequency Cepstral Coefficients (MFCCs) [11]
			- Magnitude spectra projected to a reduced set of frequency bands converted to logarithmic magnitudes
			- Approximately whitened and compressed with a discrete cosine transform (DCT).
			- Log-mel spectrum
			- The mel filter bank for projecting frequencies > It was inspired by the human auditory system and physiological findings on speech perception [12]
			- The constant-Q spectrum achieves such a frequency scale with a suitable filter bank [13]
			- Harmonics > To allow a spatially local model (e.g., a CNN) to take these into account, a third dimension can be added that directly yields the magnitudes of the harmonic series [14], [15]. 
			- The window size 
				- For computing spectra trades temporal resolution (short windows) against frequential resolution (long windows). 
				- Alternatives include computing spectra with different window lengths, projected down to the same frequency bands, and treated as separate channels [16]. In [17] the authors also investigated combinations of different spectral features.
			- Data-driven statistical model learning
				- [18] and [19] use a full-resolution magnitude spectrum
				- [20]–[23] directly use a raw waveform and learn data- driven filters jointly with the rest of the network for the target tasks
				- In [24], the lower layers of the model are designed to mimic the log-mel spectrum computation but with all the filter parameters learned from the data.
				- In [25], the notion of a filter bank is discarded, learning a causal regression model of the time-domain waveform samples without any human prior knowledge.
			- The activations at lower layers of DNNs can be thought of as speaker-adapted features, while the activations of the upper layers of DNNs can be thought of as performing class-based discrimination. ,Mohamed et al. [10]
		3. Model
			1. CNN, Convolutional Neural Networks
				- For raw waveform inputs with a high sample rate, reaching a sufficient receptive field size may result in a large number of parameters of the CNN and high computational complexity.
				- Alternatively, a dilated convolution (also called atrous, or convolution with holes) [25], [27]–[29]
				- Optimal CNN architecture condition: size of kernels, pooling and feature maps, number of channels and consecutive layers
				- a given task are not available at the time of writing (see also [30])
				- the architecture of a CNN is largely chosen experimentally based on a validation error, which has led to some rule-of-thumb guidelines, such as fewer parameters for less data [31]
			2. RNN
				- RNNs follow a different ap- proach for modeling sequences [32]
				- For offline[What is the meaning of offline?] applications, bidirectional RNNs
				with linear growth of the number of recurrent hidden units in RNNs with all-to-all kernels, the number of representable states grows exponentially, whereas training or inference time grows only quadratically at most [33]. 
				- Long short term memory (LSTM) [7] to mitigate the information flow and alleviate gradient problems.
					- Stacking of recurrent layers [34] and sparse recurrent networks [35]
					- They do not require pooling operations and are more adaptable to a range of types of input features.
					- Frequency LSTMs (F-LSTM) [36]
					- Time-Frequency LSTMs (TF-LSTM) [37]–[39] 
					- TF-LSTMs outperform CNNs on certain tasks [39], but are less parallelizable and therefore slower.
				- Convolutional Recurrent Neural Network (CRNN).
			3. Sequence-to-Sequence 
				- Traditional ASR systems comprise separate acoustic, pronunciation, and language modeling components that are normally trained independently [40], [41].
				- Directly map the input audio signal to the target sequences [42]–[47].
				- These systems are trained to optimize criteria that are related to the final evaluation metric (such as word error rate for ASR systems)
				- This greatly simplifies training compared to conventional systems: it does not require bootstrapping from decision trees or time alignments generated from a separate system, and the process of decoding is also simplified.
				- The connectionist temporal classification (CTC) [48]–[51]
				- Extended by Graves [42] to include a separate recurrent language model component, referred to as the recurrent neural network transducer (RNN- T).
				- Attention-based models[43], [52], [53]
					- The input and output sequences jointly with the target optimization
					- listen, attend and spell (LAS)[54]
			4. GANs
				- learn to produce realistic samples of a given dataset from low-dimensional, random latent vectors [55]. 
				- a generator and a discriminator. The generator maps latent vectors drawn from some known prior to samples and the discriminator is tasked with determining if a given sample is real or fake
				- The success of GANs for image synthesis[55]
				- Their use in the audio domain has been limited.
				- Source separation [56]
				- Music instrument transformation [57] 
				- Speech enhancement to transform noisy speech input to denoised versions [58]–[61]
			5. Loss Function
				- The mean squared error (MSE) between log-mel spectra can be used to quantify the difference between two frames of audio in terms of their spectral envelopes. 
				- To account for the temporal structure, log- mel spectrograms can be compared.
				- Phase issue
					- Comparing two audio signals by taking the MSE between the samples in the time domain is not a robust measure. For example, the loss for two sinusoidal signals with the same frequency would entirely depend on the difference between their phases.
					- To account for the fact that slightly non-linearly warped signals sound similar, differentiable dynamic time warping distance [62] or earth mover’s distance such as in Wasserstein GANs [63] might be more suitable.
			6. Phase modeling
				- In the calculation of the log-mel spectrum, the magnitude spectrum is used but the phase spec- trum is lost.
				- The phase can be estimated from the magnitude spectrum using the Griffin-Lim Algorithm [65].
				- A neural network (e.g. WaveNet [25]) can be trained to generate a time- domain signal from log-mel spectra [66]. Alternatively, deep learning architectures may be trained to ingest the complex spectrum directly by including both magnitude and phase spectrum as input features [67] or via complex targets [68]; alternatively all operations (convolution, pooling, activation functions) in a DNN may be extended to the complex domain [69].
				- When using raw waveform as input representation, for an analysis task, one of the difficulties is that perceptually and semantically identical sounds may appear at distinct phase shifts, so using a representation that is invariant to small phase shifts is critical
					- To achieve phase invariance researchers have usually used convolutional layers which pool in time [20], [21], [23] or DNN layers with large, potentially overcomplete, hidden units [22], which are able to capture the same filter shape at a variety of phases. 
				- Raw audio as input representation is often used in synthesis tasks, e.g. when autoregressive models are used [25].
			7. Currently, attention/transformer/DCT models are comming
		4. Data
			1. Speech Recognition 
				There are large datasets [71], for English in particular. For music se- quence classification or music similarity, there is the Million Song Dataset [72], whereas MusicNet [73] addresses note-by- note sequence labeling. Datasets for higher-level musical se- quence labeling, such as chord, beat, or structural analysis are often much smaller [74]. For environmental sound sequence classification, the AudioSet [9] of more than 2 million audio snippets is available.
			2. Transfer learning
				For example, deep neural networks trained on the ImageNet dataset can be adapted to other classification problems using small amounts of task-specific data by retraining the last layers or finetuning the weights with a small learning rate. In speech recognition, a model can be pretrained on languages with more transcribed data and then adapted to a low-resource language [75] or domains [76].
			3. Data generation and Augmentation
				The performance of an algorithm on real data may be poor if trained on generated data only. Data augmentation generates additional training data by manipulating existing examples to cover a wider range of possible inputs. 
				For ASR, [77] and [78] independently proposed to transform speech excerpts by pitch shifting (termed vocal tract perturbation) and time stretching. For far-field ASR, single-channel speech data can be passed through room simulators to generate multi-channel noisy and reverberant speech [79].
					chord recognition [80]
					time stretching and spectral filtering for singing voice detection [81]
					instrument recognition [82]
				For environmental sounds, linearly combining training examples along with their labels improves generalization [83]
		5. Evaluation
			1. Speech Recognition: WER(word error rate)
			2. Scene classification: AUROC(The area under the receiver operating charteristic)
			3. Evene detection: equal error rate or F-score
				- The true positives, false positives and false negatives are calculated either in fixed-length segments or per event [84], [85]
			4. Objective soure separation quality
				- Signal-to-distortion ratiosignal-to-interference ratio, and signal-to-artifacts ratio [86]. 
				- The mean opinion score (MOS) is a subjective test for evaluating quality of synthesized audio, in particular speech.
	- Application
		1. Analysis
			1. Speech
				- For decades, the triphone-state Gaussian mixture model (GMM) / hidden Markov model (HMM)
				- Around 1990, discriminative training was found to yield better performance than models trained using maximum likelihood. Neural network based hybrid models.
				- In 2012, DNNs on various speech recognition tasks [3]
				- In addition to the great success of deep feedforward and convolutional networks [91], LSTMs and GRUs have been shown to outperform feedforward DNNs [92]
				- Later, a cascade of convolutional, LSTM and feedforward layers, i.e. the convolutional, long short-term memory deep neural network (CLDNN) model, was further shown to outperform LSTM-only models [93]
				- With the adoption of RNNs for speech modeling, The research field shifted towards full sequence-to-sequence models. 
				- Learning a purely neural sequence-to-sequence model, such as CTC and LAS.
				- In [45], Soltau et al. trained a CTC-based model with word output targets, which was shown to outperform a state-of-the-art CD-phoneme baseline on a YouTube video captioning task. 
				- The listen, attend and spell (LAS) model is a single neural network that includes an encoder which is analogous to a conventional acoustic model, an attention module that acts as an alignment model, and a decoder that is analogous to the language model in a conventional system. 
				- Despite the architectural simplicity and empirical performance of such sequence-to-sequence models, further improvements in both model structure and optimization process have been proposed to outperform conventional models [94].
				- Voice Activity Detection [95], speaker recognition [96], language recognition [97] and speech translation [98].
			2. Music 
				1. Tasks by anaysis
					- [99] for a more extensive list.
					- Low-level analysis 
						- onset and offset detection
						- fundamental frequency estimation
					- Rhythm analysis 
						- beat tracking
						- meter identification
						- downbeat tracking
						- tempo estimation
					- Harmonic analysis 
						- key detection
						- melody extraction
						- chord estimation
					- High-level analysis 
						- instrument detection
						- instrument separation
						- transcription
						- structural segmentation
						- artist recognition
						- genre classification
						- mood classification
					- High-level comparison 
						- discovery of repeated themes
						- cover song identification
						- music similarity estimation
						- score alignment
				2. Problems
					- binary event detection problems
						- onset detection
							
							- Predicting which positions in a recording are starting points of musically relevant events such as notes, without further categorization.
							
							- The first application of neural networks to music audio: In 2006, Lacoste and Eck [84] trained a small MLP on 200 ms-excerpts of a constant-Q log-magnitude spectrogram to predict whether there is an onset in or near the center. 
							
							- Eyben et al. [100] improved over this method, applying a bidirectional LSTM to spectrograms processed with a time difference filter, albeit using a larger dataset for training. 
							
							- Schlüter et al. [16] further improved results with a CNN processing 15-frame log-mel excerpts of the same dataset.
						- beat tracking
							
							- Onset detection used to form the basis for beat and downbeat tracking [101]
							
							- Durand et al. [102] apply CNNs and Böck et al. [103] train an RNN on spectrograms to directly track beats and downbeats. Both studies rely on additional post-processing with a temporal model ensuring longer-term coherence than captured by the networks, either in the form of an HMM [102] or Dynamic Bayesian Network (DBN) [103].
							
							- Fuentes et al. [104] propose a CRNN that does not require post-processing, but also relies on a beat tracker. A higher- level event detection task is to predict boundaries between musical segments.
							
							- Ullrich et al. [105] solved it with a CNN, using a receptive field of up to 60 s on strongly downsampled spectrograms

							- For the former, it seems critical to blur training targets in time [16], [84], [105].
					- multi-class sequence labelling problem
						- chord recognition
							
							- The task of assigning each time step in a (Western) music recording a root note and chord class. 

							- Typical: folding multiple octaves of a spectral representation into a 12-semitone chromagram [13], smoothing in time, and matching against predefined chord templates.

							- Humphrey and Bello [80] note the resemblance to the operations of a CNN, and demonstrate good performance with a CNN trained on constant-Q, linear-magnitude spectrograms preprocessed with contrast normalization and augmented with pitch shifting.

							- Temporal modelling, and extend the set of distinguishable chords.

								- McFee and Bello [106] apply a CRNN (a 2D convolution learning spectrotemporal features, followed by a 1D convolution integrating information across frequencies, followed by a bidirectional GRU) and use side targets to incorporate relationships between a detailed set of 170 chord classes.

								- Korzeniowski et al. [107] train CNNs on log-frequency spectrograms to not only predict chords
					- To estimate the global tempo of a piece
						- Base it on beat and downbeat tracking
							
							- Downbeat tracking may integrate tempo estimation to constrain downbeat positions [102], [103].

							- As beat tracking can be done without onset detection, Schreiber and Müller [108] showed that CNNs can be trained to directly estimate the tempo from 12-second spectrogram excerpts,
						- Tag prediction
							
							- Aims to predict which labels from a restricted vocabulary users would attach to a given music piece

							- Tags can refer to the instrumentation, tempo, genre, and others, but always apply to a full recording, without timing information.

							- Dieleman et al. [109] train a CNN with short 1D convolutions (i.e., convolving over time only) on 3-second log-mel spectrograms, and averaged predictions over consecutive excerpts to obtain a global label.

							- Choi et al. [110] use a FCN of 3 × 3 convolutions interleaved with max-pooling such that a 29-second log-mel spectrogram is reduced to a 1×1 feature map and classified.
							Compared to FCNs in computer vision which employ average pooling in later layers of the network, max-pooling was chosen to ensure that local detections of vocals are elevated to global predictions.

							- Lee et al. [111] train a CNN on raw samples, using only short filters (size 2 to 4) interleaved with max-pooling, matching the performance of log-mel spectrograms. Like Dieleman et al., they train on 3-second excerpts and average predictions at test time.
			3. Environmental Sounds
				1. Acoustic scene classification
				2. Acoustic event detection
					
					- A simple way to do so is to concatenate acoustic features from multiple context frames around the target frame, as done in the baseline method for the public DCASE (Detection and Classification of Acoustic Events and Scenes) evaluation campaign in 2016 [112]. 

					- ALternatively, classifier architectures which model temporal information may be used: for example, recurrent neural networks may be applied to map a sequence of frame-wise acoustic features to a sequence of binary vectors representing event class activities [113]

					- In order to be able to output an event activity vector at a sufficiently high temporal resolution, the degree of max pooling or stride over time should not be too large – if a large receptive field is desired, dilated convolution and dilated pooling can be used instead [114].
				3. Tagging, polyphonic event detection
			4. Localization and Tracking
				
				- Given a microphone array signal from multiple microphones, direction estimation can be formulated in two ways: 1) by forming a fixed grid of possible directions, and by using multilabel classification to predict if there is an active source in a specific direction [115], or 2) by using regression to predict the directions [116] or spatial coordinates [117] of target sources.

				- input features
					phase spectrum [115], magnitude spectrum [118], and generalized cross-correlation between channels [117].

				- In general, source localization re- quires the use of interchannel information, which can also be learned by a deep neural network with a suitable topology from within-channel features, for example by convolutional layers [118] where the kernels span multiple channels.
		2. Synthesis and Transformation	
			1. Source Separation					
				- Masking operations in the time-frequency domain (even though there are approaches that operate directly on time-domain signals and use a DNN to learn a suitable representation from it, see e.g. [119])

				- The reason for time-frequency processing stems mainly from three factors 
					1. the structure of natural sound sources is more prominent in the time-frequency domain, which allows modeling them more easily than time-domain signals
					2. convolutional mixing which involves an acoustic transfer function from a source to a microphone which can be approximated as instantaneous mixing in the frequency domain, simplifying the processing
					3. natural sound sources are sparse in the time-frequency domain which facilitates their separation in that domain.

				- The use of these(constant-Q or mel spectrograms) has however become less common since they reduce output quality, and deep learning does not require a compact input representation that they would provide in comparison to the STFT.

				- One microphone rely on modeling the spectral structure of sources
					- Aim to predict the separation mask Mi(f,t) based on the mixture input X(f,t)
						- Deep learning in these cases is based on supervised learning based on the relation between the input mixture spectrum X(f,t) and the target output as either the oracle mask or the clean signal spectrum [120]. The oracle mask takes either binary values, or continuous values between 0 and 1.
						- convolutional [121] and recurrent [122] layers.
						- The conventional mean-square error loss is not optimal for subjective separation quality, and therefore custom loss functions have been developed to improve intelligibility [123].
					- deep clustering [124]
						- To estimate embedding vectors for each time-frequency point, which are then clustered in an unsupervised manner
						- This approach allows separation of sources that were not present in the training set. This approach can be further extended to a deep attractor network, which is based on estimating a single attractor vector for each source, and has been used to obtain state-of-the-art results in single- channel source separation [125].

				- multiple audio channels e.g. captured by multiple microphones
					- a similar manner to single-channel methods, i.e. to model the single-channel spectrum or the separation mask of a target source [126]
					- In the case of multichannel audio, the input features to a deep neural network can include spatial features in addition to spectral features (e.g. [127]). Furthermore, DNNs can be used to estimate the weights of a multi-channel mask (i.e., a beamformer) [128].

				- Regarding the different audio domains, in speech it is assumed that the signal is sparse and that different sources are independent from each other. In environmental sounds, independence can usually be assumed. In music there is a high dependence between simultaneous sources as well as there are specific temporal dependencies across time, in the waveform as well as regarding long-term structural repetitions.
			
			2. Audio Enhancement
				- They are crucial components, either explicitly [129] or implicitly [130], [131], in ASR systems for noise robustness. 
				- [129]: conventional enhancement techniques
				- Deep neural networks have been widely adopted to either directly reconstruct clean speech [132], [133] or estimate masks [134]–[136] from the noisy signals.
				- Denoising autoencoders [137]
				- Convolutional networks [121]
				- Recurrent networks [138]
				- GANs have been shown to perform well in speech enhancement in the presence of additive noise [58], when enhancement is posed as a translation task from noisy signals to clean ones.
				- The proposed speech enhancement GAN (SEGAN)
				- In [59], GANs are used to enhance speech represented as log- mel spectra. When GAN-enhanced speech is used for ASR, no improvement is found compared to enhancement using a simpler regression approach.
			
			3. Generative Models	
				1. Generating sound	
					
					- The generated sound should be original, i.e. it should be significantly different from sounds in the training set, instead of simply copying training set sounds.
					
					- The generated sounds should show diversity. 
					
					- Training and generation time should be small; ideally generation should be possible in real-time.
				2. Sound Synthesis
					
					- Performed based on a spectral representation (e.g. log-mel spectrograms) or from raw audio
					
					- The former representation lacks the phase information that needs to be reconstructed in the synthesis, e.g. via the Griffin-Lim algorithm [65] in combination with the inverse Fourier transform [139] which does not reach high synthesis quality.
					
					- End-to-end synthesis may be performed block-wise or with an autoregressive model, where sound is generated sample-by-sample, each new sample conditioned on previous samples.
					
					- In the blockwise approach, in the case of variational autoencoder (VAE) or GANs [140], the sound is often synthesised from a low-dimensional latent representation, from which it needs to by upsampled (e.g. through nearest neighbor or linear interpolation) to the high resolution sound. 
					
					- Artifacts, induced by the different layer resolutions, can be ameliorated through random phase perturbation in different layers [140].
					
					- In the autoregressive approach, the new samples are synthesised iteratively, based on an infinitely long context of previous samples, when using RNNs (such as LSTM or GRU), at the cost of expensive computation when training.
					
					- However, layers of RNNs may be stacked to process the sound on different temporal resolutions, where the activations of one layer depend on the activations of the next layer with coarser resolution [34].
					
					- An efficient audio generation model [35] based on sparse RNNs folds long sequences into a batch of shorter ones. 
					
					- Stacking dilated convolutions in the WaveNet [25] can lead to context windows of reasonable size. Using WaveNet [25], the autoregressive sample prediction is cast as a classification problem, the amplitude of the predicted sample being quantized logarithmically into distinct classes, each corresponding to an interval of amplitudes. Containing the samples, the input can be extended with context information [25]. This context may be global (such as a speaker identity) or changing during time (such as f0 or mel spectra) [25]. 

					- In [66], a text-to-speech system is introduced which consists of two modules: (1) a neural network is trained from textual input to predict a sequence of mel spectra, used as contextual input to (2) a WaveNet yielding synthesised speech. WaveNet- based models for speech synthesis outperform state-of-the-art systems by a large margin, but their training is computationally expensive. The development of parallel WaveNet [141] provides a solution to the slow training problem and hence speeds up the adoption of WaveNet models in other applications [66], [142], [143]. 

					- In [144], synthesis is controlled through parameters in the latent space of an autoencoder, applied e.g. to morph between different instrument timbres. Briot et al. [145] provide a more in-depth treatment of music generation with deep learning.
				3. Evaluation
					- Recognizability of generated sounds can be tested objectively through a classifier(e.g. inception score in [140]) 
					- Subjectively in a forced choice test with humans
					- Sounds being represented as normalized log-mel spectra, diversity can be measured as the average Euclidean distance between the sounds and their nearest neighbors.
					- Originality can be measured as the average Euclidean distance between a generated samples to their near- est neighbor in the real training set [140].
					- Turing test
	- Discussion
		1. Features
		2. Models
		3. Data Requirements
		4. Computation Complexity
		5. Interpretability and Adaptability

- A Comparative Study of Time and Frequency Domain Approaches to Deep Learning based Speech Enhancement, 2020
	Time vs Frequency domain approach https://ieeexplore.ieee.org/abstract/document/9206928
	- Comparision DNN with learing time, frequency or both sides domain
	- Feature based learning has been proved to positively affect the learning process [1]
	- Refernece
		[1] L. Hertel, H. Phan, and A. Mertins, “Comparing time and frequency domain for audio event recognition using deep learning,” in IJCNN. IEEE, 2016, pp. 3407–3411.
- [ASR] Deep Learning for Environmentally Robust Speech Recognition: An Overview of Recent Developments, Zixing Zhang, 2017 [[paper]](https://arxiv.org/pdf/1705.10874.pdf)

- [Speech Speration]Supervised speech separation based on deep learning: An Overview, 2017 [[paper]](https://arxiv.org/pdf/1708.07524.pdf)

- [Classical method]Nonlinear speech enhancement: an overview, 2007 [[paper]](https://www.researchgate.net/publication/225400856_Nonlinear_Speech_Enhancement_An_Overview)

#### 2.2 Magnitude spectrum
- spectral masking
	- 2014, On Training Targets for Supervised Speech Separation, Wang. [[Paper]](https://ieeexplore.ieee.org/document/6887314)  
	- 2018, A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement, [Valin](https://github.com/jmvalin). [[Paper]](https://ieeexplore.ieee.org/document/8547084/) [[RNNoise]](https://github.com/xiph/rnnoise) [[RNNoise16k]](https://github.com/YongyuG/rnnoise_16k)

	- 2020, A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech, [Valin](https://github.com/jmvalin). [Paper](https://arxiv.org/abs/2008.04259) [[PercepNet]](https://github.com/jzi040941/PercepNet)
	- 2020, Online Monaural Speech Enhancement using Delayed Subband LSTM, Li. [[Paper]](https://arxiv.org/abs/2005.05037)
	- 2020, FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement, [Hao](https://github.com/haoxiangsnr). [[Paper]](https://arxiv.org/pdf/2010.15508.pdf) [[FullSubNet]](https://github.com/haoxiangsnr/FullSubNet)
	- 2021, RNNoise-Ex: Hybrid Speech Enhancement System based on RNN and Spectral Features. [[Paper]](https://arxiv.org/abs/2105.11813) [[RNNoise-Ex]](https://github.com/CedArctic/rnnoise-ex)
	- Other IRM-based SE repositories: [[IRM-SE-LSTM]](https://github.com/haoxiangsnr/IRM-based-Speech-Enhancement-using-LSTM) [[nn-irm]](https://github.com/zhaoforever/nn-irm) [[rnn-se]](https://github.com/amaas/rnn-speech-denoising) [[DL4SE]](https://github.com/miralv/Deep-Learning-for-Speech-Enhancement)

- spectral mapping
	- 2014, An Experimental Study on Speech Enhancement Based on Deep Neural Networks, [Xu](https://github.com/yongxuUSTC). [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6665000)

	- 2014, A Regression Approach to Speech Enhancement Based on Deep Neural Networks, [Xu](https://github.com/yongxuUSTC). [[Paper]](https://ieeexplore.ieee.org/document/6932438) [[sednn]](https://github.com/yongxuUSTC/sednn) [[DNN-SE-Xu]](https://github.com/yongxuUSTC/DNN-Speech-enhancement-demo-tool) [[DNN-SE-Li]](https://github.com/hyli666/DNN-SpeechEnhancement) 

	- Other DNN magnitude spectrum mapping-based SE repositories: [[SE toolkit]](https://github.com/jtkim-kaist/Speech-enhancement) [[TensorFlow-SE]](https://github.com/linan2/TensorFlow-speech-enhancement-Chinese) [[UNetSE]](https://github.com/vbelz/Speech-enhancement)

	- 2015, Speech enhancement with LSTM recurrent neural networks and its application to noise-robust ASR, Weninger. [[Paper]](https://hal.inria.fr/hal-01163493/file/weninger_LVA15.pdf)

	- 2016, A Fully Convolutional Neural Network for Speech Enhancement, Park. [[Paper]](https://arxiv.org/abs/1609.07132) [[CNN4SE]](https://github.com/dtx525942103/CNN-for-single-channel-speech-enhancement)

	- 2017, Long short-term memory for speaker generalizationin supervised speech separation, Chen. [[Paper]](http://web.cse.ohio-state.edu/~wang.77/papers/Chen-Wang.jasa17.pdf)

	- 2018, A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement, [Tan](https://github.com/JupiterEthan). [[Paper]](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf) [[CRN-Tan]](https://github.com/JupiterEthan/CRN-causal)

	- 2018, Convolutional-Recurrent Neural Networks for Speech Enhancement, Zhao. [[Paper]](https://arxiv.org/pdf/1805.00579.pdf) [[CRN-Hao]](https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement)	

#### 2.3 Complex domain
- 2017, Complex spectrogram enhancement by convolutional neural network with multi-metrics learning, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/pdf/1704.08504.pdf)
- 2017, Time-Frequency Masking in the Complex Domain for Speech Dereverberation and Denoising, Williamson. [[Paper]](https://ieeexplore.ieee.org/abstract/document/7906509)
- 2019, PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network, Yin. [[Paper]](https://arxiv.org/abs/1911.04697) [[PHASEN]](https://github.com/huyanxin/phasen)
- 2019, Phase-aware Speech Enhancement with Deep Complex U-Net, Choi. [[Paper]](https://arxiv.org/abs/1903.03107) [[DC-UNet]](https://github.com/chanil1218/DCUnet.pytorch)
- 2020, Learning Complex Spectral Mapping With GatedConvolutional Recurrent Networks forMonaural Speech Enhancement, [Tan](https://github.com/JupiterEthan). [[Paper]](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang.taslp20.pdf) [[GCRN]](https://github.com/JupiterEthan/GCRN-complex)
- 2020, DCCRN: Deep Complex Convolution Recurrent Network for Phase-AwareSpeech Enhancement, [Hu](https://github.com/huyanxin). [[Paper]](https://isca-speech.org/archive/Interspeech_2020/pdfs/2537.pdf) [[DCCRN]](https://github.com/huyanxin/DeepComplexCRN)
- 2020, T-GSA: Transformer with Gaussian-Weighted Self-Attention for Speech Enhancement, Kim. [[Paper]](https://ieeexplore.ieee.org/document/9053591) 
- 2020, Phase-aware Single-stage Speech Denoising and Dereverberation with U-Net, Choi. [[Paper]](https://arxiv.org/abs/2006.00687)
- 2021, DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement, [Le](https://github.com/Le-Xiaohuai-speech). [[Paper]](https://www.isca-speech.org/archive/pdfs/interspeech_2021/le21b_interspeech.pdf) [[DPCRN]](https://github.com/Le-Xiaohuai-speech/DPCRN_DNS3)
- 2021, Real-time denoising and dereverberation with tiny recurrent u-net, Choi. [[Paper]](https://arxiv.org/pdf/2102.03207.pdf)
- 2021, DCCRN+: Channel-wise Subband DCCRN with SNR Estimation for Speech Enhancement, [Lv](https://github.com/IMYBo/)  [[Paper]](https://arxiv.org/abs/2106.08672)

#### 2.4 Time domian
- 2018, Improved Speech Enhancement with the Wave-U-Net, Macartney. [[Paper]](https://arxiv.org/pdf/1811.11307.pdf) [[WaveUNet]](https://github.com/YosukeSugiura/Wave-U-Net-for-Speech-Enhancement-NNabla) 
- 2019, A New Framework for CNN-Based Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/8701652) 
- 2019, TCNN: Temporal Convolutional Neural Network for Real-time Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/8683634)
- 2020, Real Time Speech Enhancement in the Waveform Domain, Defossez. [[Paper]](https://arxiv.org/abs/2006.12847) [[facebookDenoiser]](https://github.com/facebookresearch/denoiser)
- 2020, Monaural speech enhancement through deep wave-U-net, Guimarães. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0957417420304061) [[SEWUNet]](https://github.com/Hguimaraes/SEWUNet) 
- 2020, Speech Enhancement Using Dilated Wave-U-Net: an Experimental Analysis, Ali. [[Paper]](https://ieeexplore.ieee.org/document/9211072)
- 2020, Densely Connected Neural Network with Dilated Convolutions for Real-Time Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ashutosh620.github.io/files/DDAEC_ICASSP_2020.pdf) [[DDAEC]](https://github.com/ashutosh620/DDAEC)
- 2021, Dense CNN With Self-Attention for Time-Domain Speech Enhancement, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/9372863)
- 2021, Dual-path Self-Attention RNN for Real-Time Speech Enhancement, [Pandey](https://github.com/ashutosh620). [[Paper]](https://arxiv.org/abs/2010.12713)

#### 2.5 GAN
- 2017, SEGAN: Speech Enhancement Generative Adversarial Network, Pascual. [[Paper]](https://arxiv.org/pdf/1703.09452.pdfsegan_pytorch) [[SEGAN]](https://github.com/santi-pdp/segan_pytorch)
- 2019, SERGAN: Speech enhancement using relativistic generative adversarial networks with gradient penalty, [Deepak Baby]((https://github.com/deepakbaby)). [[Paper]](https://biblio.ugent.be/publication/8613639/file/8646769.pdf) [[SERGAN]](https://github.com/deepakbaby/se_relativisticgan)
- 2019, MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/pdf/1905.04874.pdf) [[MetricGAN]](https://github.com/JasonSWFu/MetricGAN)
- 2019, MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/abs/2104.03538) [[MetricGAN+]](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN)
- 2020, HiFi-GAN: High-Fidelity Denoising and Dereverberation Based on Speech Deep Features in Adversarial Networks, Su. [[Paper]](https://arxiv.org/abs/2006.05694) [[HifiGAN]](https://github.com/rishikksh20/hifigan-denoiser)

#### 2.6 Hybrid SE
- 2019, Deep Xi as a Front-End for Robust Automatic Speech Recognition, [Nicolson](https://github.com/anicolson). [[Paper]](https://arxiv.org/abs/1906.07319) [[DeepXi]](https://github.com/anicolson/DeepXi)

- 2019, Using Generalized Gaussian Distributions to Improve Regression Error Modeling for Deep-Learning-Based Speech Enhancement, [Li](https://github.com/LiChaiUSTC). [[Paper]](http://staff.ustc.edu.cn/~jundu/Publications/publications/chaili2019trans.pdf) [[SE-MLC]](https://github.com/LiChaiUSTC/Speech-enhancement-based-on-a-maximum-likelihood-criterion)

- 2020, Deep Residual-Dense Lattice Network for Speech Enhancement, [Nikzad](https://github.com/nick-nikzad). [[Paper]](https://arxiv.org/pdf/2002.12794.pdf) [[RDL-SE]](https://github.com/nick-nikzad/RDL-SE)

- 2020, DeepMMSE: A Deep Learning Approach to MMSE-based Noise Power Spectral Density Estimation, [Zhang](https://github.com/yunzqq). [[Paper]](https://ieeexplore.ieee.org/document/9066933)

- 2020, Speech Enhancement Using a DNN-Augmented Colored-Noise Kalman Filter, [Yu](https://github.com/Hongjiang-Yu). [[Paper]](https://www.sciencedirect.com/science/article/pii/S0167639320302831) [[DNN-Kalman]](https://github.com/Hongjiang-Yu/DNN_Kalman_Filter)

#### 2.7 Multi-stage
- 2020, A Recursive Network with Dynamic Attention for Monaural Speech Enhancement, [Li](https://github.com/Andong-Li-speech). [[Paper]](https://arxiv.org/abs/2003.12973) [[DARCN]](https://github.com/Andong-Li-speech/DARCN)

- 2020, Masking and Inpainting: A Two-Stage Speech Enhancement Approach for Low SNR and Non-Stationary Noise, [Hao](https://github.com/haoxiangsnr). [[Paper]](https://ieeexplore.ieee.org/document/9053188/)

- 2020, A Joint Framework of Denoising Autoencoder and Generative Vocoder for Monaural Speech Enhancement, Du. [[Paper]](https://ieeexplore.ieee.org/document/9082858)

- 2020, Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression, [Westhausen](https://github.com/breizhn). [[Paper]](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf) [[DTLN]](https://github.com/breizhn/DTLN)

- 2020, Listening to Sounds of Silence for Speech Denoising, [Xu](https://github.com/henryxrl). [[Paper]](http://www.cs.columbia.edu/cg/listen_to_the_silence/paper.pdf) [[LSS]](https://github.com/henryxrl/Listening-to-Sound-of-Silence-for-Speech-Denoising)

- 2021, ICASSP 2021 Deep Noise Suppression Challenge: Decoupling Magnitude and Phase Optimization with a Two-Stage Deep Network, [Li](https://github.com/Andong-Li-speech). [[Paper]](https://arxiv.org/abs/2102.04198)

- 2022, Glance and Gaze: A Collaborative Learning Framework for Single-channel Speech Enhancement, [Li](https://github.com/Andong-Li-speech/GaGNet) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0003682X21005934)

- 2022, HGCN : harmonic gated compensation network for speech enhancement, [Wang](https://github.com/wangtianrui/HGCN). [[Paper]](https://arxiv.org/pdf/2201.12755.pdf)	

#### 2.8 Feature augmentation
- Speech enhancement using self-adaptation and multi-head attention, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.05873.pdf)

- PAN: phoneme-aware network for monaural speech enhancement, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9054334)

- Noise tokens: learning neural noise templates for environment-aware speech enhancement [[paper]](https://arxiv.org/pdf/2004.04001.pdf)

- Speaker-aware deep denoising autoencoder with embedded speaker identity for speech enhancement, Interspeech 2019 [[paper]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2108.pdf)

#### 2.9 Audio-Visual SE 
- Lite Audio-Visual Speech Enhancement, INTERSPEECH 2020 [[paper]](https://arxiv.org/pdf/2005.11769.pdf)

- Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks, TETCI, 2018 (first audio-visual SE) [[journal]](https://ieeexplore.ieee.org/document/8323326)

#### 2.10 Netword design
- Filter design
	- Efficient trainable front-ends for neural speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.09286.pdf)

- Fusion technique
	- Spectrograms fusion with minimum difference masks estimation for monaural speech dereverberation, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/abstract/document/9054661)

	- Masking and inpainting: a two-stage speech enhancement approach for low snr and non-stationary noise, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9053188)

	- A composite dnn architecture for speech enhancement, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9053821)

	- An attention-based neural network approach for single channel speech enhancement, ICASSP 2019 [[paper]](http://www.npu-aslp.org/lxie/papers/2019ICASSP-XiangHao.pdf)

	- Multi-domain processing via hybrid denoising networks for speech enhancement, 2018 [[paper]](https://arxiv.org/pdf/1812.08914.pdf)

- Attention
	- Speech enhancement using self-adaptation and multi-head attention, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.05873.pdf)

	- Channel-attention dense u-net for multichannel speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2001.11542.pdf)

	- T-GSA: transformer with gaussian-weighted self-attention for speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/1910.06762.pdf)

- U-Net
	- Phase-aware speech enhancement with deep complex u-net, ICLR 2019 [[paper]](https://openreview.net/pdf?id=SkeRTsAcYm) [[code]](https://github.com/sweetcocoa/DeepComplexUNetPyTorch)
- GAN
	- PAGAN: a phase-adapted generative adversarial networks for speech enhancement, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9054256) 
	
	- MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement, ICML 2019 [[paper]](http://proceedings.mlr.press/v97/fu19b/fu19b.pdf)
	
	- Time-frequency masking-based speech enhancement using generative adversarial network, ICASSP 2018 [[paper]](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005039.pdf)

	- SEGAN: speech enhancement generative adversarial network, Interspeech 2017 [[paper]](https://arxiv.org/pdf/1703.09452.pdf) 
- Auto-Encoder
	- Speech Enhancement Based on Deep Denoising Autoencoder, INTERSPEECH 2013 (first deep learning based SE) [[paper]](https://www.citi.sinica.edu.tw/papers/yu.tsao/3582-F.pdf)

#### 2.11 Phase reconstruction
- Phase reconstruction based on recurrent phase unwrapping with deep neural networks, ICASSP 2020 [[paper]](https://arxiv.org/pdf/2002.05832.pdf)

- PAGAN: a phase-adapted generative adversarial networks for speech enhancement, ICASSP 2020 [[paper](https://ieeexplore.ieee.org/document/9054256)

- Invertible dnn-based nonlinear time-frequency transform for speech enhancement, ICASSP 2020 [[paper]](https://arxiv.org/pdf/1911.10764.pdf)

- Phase-aware speech enhancement with deep complex u-net, ICLR 2019 [[paper]](https://openreview.net/pdf?id=SkeRTsAcYm) [[code]](https://github.com/sweetcocoa/DeepComplexUNetPyTorch)
- PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network, AAAI 2020 [[paper]](https://aaai.org/Papers/AAAI/2020GB/AAAI-YinD.3057.pdf)

#### 2.12 Learning strategy
- Loss function
	- MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement, ICML 2019 [[paper]](http://proceedings.mlr.press/v97/fu19b/fu19b.pdf)

	- Speech denoising with deep feature losses, Interspeech 2019 [[paper]](https://arxiv.org/pdf/1806.10522.pdf)

	- End-to-end multi-task denoising for joint sdr and pesq optimization, Arxiv 2019 [[paper]](https://arxiv.org/pdf/1901.09146.pdf)

- Multi-task learning
	- Multi-objective learning and mask-based post-processing for deep neural network based speech enhancement, Arxiv 2017 [[paper]](https://arxiv.org/pdf/1703.07172.pdf)

	- Multiple-target deep learning for LSTM-RNN based speech enhancement, HSCMA 2017 [[paper]](http://home.ustc.edu.cn/~sunlei17/pdf/MULTIPLE-TARGET.pdf)
	
	- Speech enhancement and recognition using multi-task learning of long short-term memory recurrent neural networks, ISCA 2015 [[paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.712.1367&rep=rep1&type=pdf)

#### 2.13 Curriculum learning
- SNR-Based Progressive Learning of Deep Neural Network for Speech Enhancement, INTERSPEECH 2016 [[paper]](http://staff.ustc.edu.cn/~jundu/Publications/publications/IS2016_Gao.pdf)

#### 2.14 Transfer learning
- Cross-language transfer learning for deep neural network based speech enhancement, ISCSLP 2014 [[paper]](http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/pdfs/Yong_ISCSLP2014.pdf)

#### 2.15 NMF
- Speech_Enhancement_DNN_NMF 
[[Code]](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF)
- gcc-nmf:Real-time GCC-NMF Blind Speech Separation and Enhancement 
[[Code]](https://github.com/seanwood/gcc-nmf)
- https://github.com/Jerry-jwz/Audio-Enhancement-via-ONMF

#### 2.16 Other improvements
- Improving robustness of deep learning based monaural speech enhancement against processing artifacts, ICASSP 2020 [[paper]](https://ieeexplore.ieee.org/document/9054145)

#### 2.17 Jointly addressing Clipping, Codec Distortions and Gaps
- https://ieeexplore.ieee.org/abstract/document/9414721

#### 2.18 Model Compression
- https://ieeexplore.ieee.org/abstract/document/9437977
- https://ieeexplore.ieee.org/abstract/document/9413536
- https://ieeexplore.ieee.org/abstract/document/8892545

#### 2.19 Performance Evaluation
- https://www.worldscientific.com/doi/abs/10.1142/S0219477519500202

#### 2.20 Masking
- human auditory masking: https://ieeexplore.ieee.org/abstract/document/748118

#### 2.21 Classical Method [17]
- Noise Classification
	- support vector machine (SVM) [1]
	- gaussian mixture model (GMM) [2]
	- Maximum A Posterior (MAP) estimation [15] 
	- Maximum Likelihood Linear Regression (MLLR) [16]
	- statistical properties of the interactions between speech and noise signals [14]

- Noise Estimation
	- minimum statistics (MS) [3]
	- minima controlled recursive averaging (MCRA) [4]
	- improved minima controlled recursive averaging (IMCRA) [5]

- Noise Reduction
	- Spectral subtraction (SS) [6]
	- Wiener filter [6, 7, 8, 13]
	- Minimum Mean Square Error Short-Time Spectral Amplitude (MMSE-STSA) [9]
	- Minimum Mean Square Error-Log Scale Amplitude (MMSE-LSA) [10]
	- Log Minimum Mean Square Error (logMMSE) [11]

- Unsupervised single-channel speech enhancement techniques 
	- Extended Kalman Filtering [28, 30, 31] 
	- Monte-Carlo simulations [27]
	- Particle filtering [27], [29], [32] 
	- Noise-Regularized Adaptive Filtering [28], [33] 

- Computational auditory scene analysis(CASA)[18], [19]
- pitch estimation and pitch-based grouping[20]
- Time-frequency (T-F) Masking[18, 21, 22]
- Ideal binary masking [23, 24, 25, 26]

- Reference for 2.21 Classical Method
	- [1] Suykens, J.A.; Vandewalle, J. Least squares support vector machine classifiers. Neural Process. Lett. 1999, 9, 293–300. 
	- [2] Kim, G.; Lu, Y.; Hu, Y.; Loizou, P.C. An algorithm that improves speech intelligibility in noise for normal-hearing listeners. J. Acoust. Soc. Am. 2009, 126, 1486–1494. 
	- [3] Martin, R. Noise power spectral density estimation based on optimal smoothing and minimum statistics. IEEE Trans. Speech Audio Process. 2001, 9, 504–512. 
	- [4] Cohen, I.; Berdugo, B. Noise estimation by minima controlled recursive averaging for robust speech enhancement. IEEE Signal Process. Lett. 2002, 9, 12–15. 
	- [5] Cohen, I. Noise spectrum estimation in adverse environments: Improved minima controlled recursive averaging. IEEE Trans. Speech Audio Process. 2003, 11, 466–475. 
	- [6] Loizou, P.C. Speech Enhancement: Theory and Practice; CRC Press: Boca Raton, FL, USA, 2013.
	- [7]  S. Boll, “Suppression of acoustic noise in speech using spectral subtraction,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 27, no. 2, pp. 113–120, Apr. 1979.
	- [8] Van den Bogaert, T.; Doclo, S.; Wouters, J.; Moonen, M. Speech enhancement with multichannel Wiener filter tehniques in multimicrophone binaural hearing aids. J. Acoust. Soc. Am. 2009, 125, 360–371. 
	- [9] Y. Ephraim and D. Malah, “Speech enhancement using a minimum- mean square error short-time spectral amplitude estimator,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 32, no. 6, pp. 1109–1121, Dec. 1984.
	- [10] Eddins, D.A. Sandlin’s Textbook of Hearing Aid Amplification; Taylor & Francis: Abingdon, UK, 2014.
	- [11] Rao, Y.; Hao, Y.; Panahi, I.M.; Kehtarnavaz, N. Smartphone-based real-time speech enhancement for improving hearing aids speech perception. In Proceedings of the 2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Orlando, FL, USA, 16–20 August 2016; pp 5885–5888.
	- [12] Ephraim, Y.; Malah, D. Speech enhancement using a minimum mean-square error log-spectral amplitude
	esimator. IEEE Trans. Acoust. Speech Signal Process. 1985, 33, 443–445. 
	- [13] J. H. Hansen and M. A. Clements, “Constrained iterative speech en- hancement with application to speech recognition,” IEEE Transaction on Signal Processing, vol. 39, no. 4, pp. 795–805, Apr. 1991.
	- [14] Y. Xu, J. Du, L. Dai, and C. Lee, “A regression approach to speech enhancement based on deep neural networks,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 1, pp. 7–19, Jan. 2015.
	- [15] J.-L. Gauvain and C.-H. Lee, “Maximum a posteriori estimation for multivariate gaussian mixture observations of markov chains,” IEEE transactions on speech and audio processing, vol. 2, no. 2, pp. 291– 298, Apr. 1994.
	- [16] C. J. Leggetter and P. C. Woodland, “Maximum likelihood linear regression for speaker adaptation of continuous density hidden markov models,” Computer Speech & Language, vol. 9, no. 2, pp. 171–185, Apr. 1995.
	- [17] J. Li, L. Deng, Y. Gong, and R. Haeb-Umbach, “An overview of noise-robust automatic speech recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 4, pp. 745– 777, Apr. 2014.
	- [18] D.L. Wang and G.J. Brown, Ed., Computational auditory scene analysis: Principles, algorithms, and applications. Hoboken NJ: Wiley & IEEE Press, 2006.
	- [19] A.S. Bregman, Auditory scene analysis. Cambridge MA: MIT Press, 1990.
	- [20] G. Hu and D.L. Wang, "A tandem algorithm for pitch estimation and voiced speech segregation," IEEE Trans.Audio Speech Lang. Proc., vol. 18, pp. 2067-2079, 2010.
	- [21] R.F. Lyon, "A computational model of binaural localization and separation," in Proceedings of ICASSP, pp. 1148-1151, 1983.
	- [22] D.L. Wang, "Time-frequency masking for speech separation and its potential for hearing aid design," Trend. Amplif., vol. 12, pp. 332-353, 2008.
	- [23] M.C. Anzalone, L. Calandruccio, K.A. Doherty, and L.H. Carney, "Determination of the potential benefit of time- frequency gain manipulation," Ear Hear., vol. 27, pp. 480- 492, 2006.
	- [24] D.S. Brungart, P.S. Chang, B.D. Simpson, and D.L. Wang,
	"Isolating the energetic component of speech-on-speech masking with ideal time-frequency segregation," J. Acoust. Soc. Am., vol. 120, pp. 4007-4018, 2006.
	- [25] N. Li and P.C. Loizou, "Factors influencing intelligibility of ideal binary-masked speech: Implications for noise reduction," J. Acoust. Soc. Am., vol. 123, pp. 1673-1682, 2008.
	- [26] D.L. Wang, U. Kjems, M.S. Pedersen, J.B. Boldt, and T. Lunner, "Speech intelligibility in background noise with ideal binary time-frequency masking," J. Acoust. Soc. Am., vol. 125, pp. 2336-2347, 2009.
	- [27] Ephraim, Y., Cohen, I.: Recent Advancements in Speech Enhancement. The Electrical Engineering Handbook, CRC Press, 2005
	- [28] Lev-Ari, H. and Ephraim, Y.: Extension of the signal subspace speech enhancement approach to colored noise. IEEE Sig. Proc. Let., vol. 10, pp. 104-106, April 2003
	- [29] Dawson, M.I. and Sridharan, S.: Speech enhancement using time delay neural networks, Proceedings of the Fourth Australian International Conf. on Speech Science and Technology, pages 152-5, December 1992
	- [30] Gannot, S., Burshtein, D. and Weinstein, E.: Iterative and Sequential Kalman Filter-Based Speech Enhancement Algorithms. IEEE Trans. Speech and Audio Proc., vol. 6, pp. 373-385, 1998
	- [31] Wan, E.A., Nelson, A.T.: Neural dual extended Kalman filtering: applications in speech enhancement and monaural blind signal separation. Proceedings Neural Networks for Signal Processing Workshop, 1997
	- [32] Fong, W., Godsill, S.J., Doucet, A. and West, M.: Monte Carlo smoothing with application to audio signal enhancement. IEEE Trans. Signal Processing, vol. 50, pp. 438-449, 2002 
	- [33] Wan, E. and Van der Merwe, R.: Noise-Regularized Adaptive Filtering for Speech Enhancement. Proceedings of EUROSPEECH’99, Sep 1999

#### 2.22 Issue
- Smearing: Spectral leakage[1] -> Solution: windowing(Hanning, Hamming)[2]

- Noise
	```
	When operating in the frequency domain it is most common to deal only with the amplitude of the speech signal, assuming that the phase is insensitive to noise [3] so the phase of the noisy speech is extracted to be added to the estimated signal when reconstructing the audio.
	However, this assumption does not always hold, as some studies show the importance of phase in improving the performance [4], [5]. Many techniques have been proposed to retrieve the clean phase, or are based on the use of the complex spectrograms in order to solve this issue [6]–[8],
	working in time results in fewer compu- tations, as the framing of the input signal is the only required operation, and some researchers even work with the waveform without the framing process. Moreover, the phase information is estimated during the training that leads to better prediction of the clean signal, and also no scaling is needed for the output signal.
	```

- Computation
	- Cost of STFT [9]
	- Scaling

- Frame size
	- be continued by Noise Part, working in the time domain results in a much higher number of network parameters due to the large frame size used, which is proved to be better than smaller frames [10], [11]. 

- Fit to Hardware, The larger number of parameters increases the size of the model, and restricts its applicability in some real time implementations as the model may not fit into the hardware [12].

- Reference for 2.22 Issue
	- [1] F.Harris,“Ontheuseofwindowsforharmonicanalysiswiththediscrete fourier transform,” Proceedings of the IEEE, vol. 66, no. 1, pp. 51–83, 1978.
	- [2] P. Podder, T. Khan, M. Khan, and M. Rahman, “Comparative performance analysis of hamming, hanning and blackman window,” Int. J. Comput. Appl., vol. 96, no. 18, 2014.
	- [3] D. Wang and J. Lim, “The unimportance of phase in speech enhancement,” IEEE Trans. Acoust. Speech Sig. Proc., vol. 30, no. 4, pp. 679–681, 1982.
	- [4] K. Paliwal, K. Wo ́jcicki, and B. Shannon, “The importance of phase in speech enhancement,” Speech Comm., vol. 53, no. 4, pp. 465–494, 2011.
	- [5] G. Shi, M. Shanechi, and P. Aarabi, “On the importance of phase in human speech recognition,” IEEE Trans. Audio Speech Lang. Proc., vol. 14, no. 5, pp. 1867–1874, 2006.
	- [6] Z. Ouyang, H. Yu, W. Zhu, and B. Champagne, “A fully convolutional neural network for complex spectrogram processing in speech enhancement,” in ICASSP. IEEE, 2019, pp. 5756–5760.
	- [7] S. Fu, T. Hu, Y. Tsao, and X. Lu, “Complex spectrogram enhancement by convolutional neural network with multi-metrics learning,” in MLSP. IEEE, 2017, pp. 1–6.
	- [8] T. Gerkmann, M. Krawczyk-Becker, and J. Le Roux, “Phase processing for single-channel speech enhancement: History and recent advances,” IEEE Sig. Proc. Mag., vol. 32, no. 2, pp. 55–66, 2015.
	- [9] S. Fu, Y. Tsao, X. Lu, and H. Kawai, “Raw waveform-based speech enhancement by fully convolutional networks,” in APSIPA ASC. IEEE, 2017, pp. 6–12.
	- [10] A. Pandey and D. Wang, “A new framework for cnn-based speech enhancement in the time domain,” IEEE Trans. Audio Speech Lang. Proc., vol. 27, no. 7, pp. 1179–1188, 2019.
	- [11] D. Eringis and G. Tamulevicˇius, “Improving speech recognition rate through analysis parameters,” J. Elect. Control and Comm. Eng., vol. 5, no. 1, pp. 61–66, 2014.
	- [12] H. Purwins, B. Li, T. Virtanen, J. Schlu ̈ter, S. Chang, and T. Sainath, “Deep learning for audio signal processing,” J. Selected Topics Sig. Proc., vol. 13, no. 2, pp. 206–219, 2019.

- Book
	- https://books.google.co.kr/books?hl=en&lr=&id=2_xlDwAAQBAJ&oi=fnd&pg=PR17&dq=speech+enhancement+survey&ots=g7tRSUtIhn&sig=EmqSEX3-cLBlyx9GzpdHr_R8tnc#v=onepage&q=speech%20enhancement%20survey&f=false

- Compression Model base
	1. TinyLSTMs- Efficient Neural Speech Enhancement for Hearing Aids

## 2. Reference
### 2.1 ML models Platform 
- https://github.com/asteroid-team/asteroid
- https://github.com/speechbrain/speechbrain
- https://github.com/facebookresearch/demucs

### 2.2 Other Research for speech enhancment methods 
- https://github.com/Wenzhe-Liu/awesome-speech-enhancement/blob/master/README.md
- https://github.com/nanahou/Awesome-Speech-Enhancement/blob/master/README.md
- https://ccrma.stanford.edu/~njb/teaching/sstutorial/part1.pdf
- https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/interspeech-tutorial-2015-lideng-sept6a.pdf
- https://www.citi.sinica.edu.tw/papers/yu.tsao/7463-F.pdf