# Datasets for Sound Source

@updated date 2022.12.31

## 1. Research for dataset amd metrics

### 1.1 Papers

1. A. Pandey and D. Wang, "Dense CNN With Self-Attention for Time-Domain Speech Enhancement," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 1270-1279, 2021, doi: 10.1109/TASLP.2021.3064421.
	
	```
	We evaluate all the models in a speaker- and noise- independent way on the WSJ0 SI-84 dataset (WSJ) [44], which consists of 7138 utterances from 83 speakers (42 males and 41 females). Seventy seven speakers are used for training and remaining six are used for evaluation. For training, we use 10 000 non-speech sounds from a sound effect library (available at www.sound-ideas.com) [9], and generate 320000 noisy utter- ances at SNRs uniformly sampled from {−5 dB, −4 dB, −3 dB, −2 dB, −1 dB, 0 dB}. For the test set, we use babble and cafeteria noises from an Auditec CD (available at http: //www.auditec.com), and generate 150 noisy utterances for both the noises at SNRs of −5 dB, 0 dB, and 5 dB.

		- Reference
		[9] J. Chen, Y. Wang, S. E. Yoho, D. L. Wang, and E. W. Healy, “Large-scale training to increase speech intelligibility for hearing-impaired listeners in novel noises,” J. Acoust. Soc. Amer., vol. 139, pp. 2604–2612, 2016.
		[44] D. B. Paul and J. M. Baker, “The design for the wall street journal-based CSR corpus,” in Proc. Workshop Speech Nat. Lang., 1992, pp. 
	```

2. H. Wang and D. Wang, "Towards Robust Speech Super-Resolution," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 2058-2066, 2021, doi: 10.1109/TASLP.2021.3054302.

	```
	TIMIT [12] and VCTK [48]
		
		- Reference
		[12] J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, and D. S. Pallett, “DARPA TIMIT acoustic-phonetic continous speech corpus CD-ROM. NIST speech disc 1-1.1,” NASA STIN, vol. 93, 1993, Art no. 27403.
		[48] C. Veaux, J. Yamagishi, and K. MacDonald, “CSTR VCTK corpus: English multi-speaker corpus for CSTR voice cloning toolkit,” Univ. Edinburgh. The Centre for Speech Technol. Res., 2017.
	```

3. H. Li, D. Wang, X. Zhang and G. Gao, "Recurrent Neural Networks and Acoustic Features for Frame-Level Signal-to-Noise Ratio Estimation," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 2878-2887, 2021, doi: 10.1109/TASLP.2021.3107617.

	```
	We evaluate the proposed algorithm on the WSJ0 SI-84 dataset [31], which includes 7138 utterances from 83 speakers (42 males and 41 females). Six (three males and three females) of these speakers are randomly selected and set aside for testing. In other words, 77 remaining speakers are used to train the model. We also hold out 150 randomly selected utterances from the 77 training speakers to create a validation set with a babble noise from the NOISEX-92 dataset [37]. We use the 10,000 noises from a sound effect library,^6 which has a total duration of about 126 hours, as the training noise set. For testing, we use six noises, i.e., babble and cafeteria noise from an Auditec CD,^7 factory and speech shape noise (SSN) from NOISEX-92, and park and traffic noise from the DEMAND noise set [36]. These test noises are selected to represent the kinds encountered in practical situations. The training set contains 100,000 mixtures, and the total duration is about 160 hours. To generate a training mixture, we mix a randomly selected training utterance and a random segment from the 10,000 training noises. The SNR is randomly sampled from -5 dB to 10 dB with a step size of 1 dB. The validation set contains 800 utterances. The SNR of the validation utterances is randomly selected from -5 dB to 10 dB with a step size of 1 dB, which is the same as in the training set. The test set includes 1,200 mixtures generated from 25 × 6 utterances of the 6 untrained speakers. The test set SNR is randomly selected from -10 dB to 15 dB with a step size of 5 dB. Note that speech and noise signals are different between training and testing, and two test SNRs are not included in the training set.
		
		- Reference
		[31] D.B.PaulandJ.M.Baker,“The design for the Wall Street Journal-based CSR corpus,” in Proc. Workshop Speech
		[36] J. Thiemann, N. Ito, and E. Vincent, “The diverse environments multi- channel acoustic noise database (DEMAND): A database of multichannel environmental noise recordings,” in Proc. Int. Congr. Acoust., pp. 1–6, 2013.
		[37] A. Varga and H. J. Steeneken, “Assessment for automatic speech recogni- tion: II. NOISEX-92: A database and an experiment to study the effect of additive noise on speech recognition systems,” Speech Commun., vol. 12, no. 3, pp. 247–251, 1993.

		^6[Online]. Available: https://www.soundideas.com 
		^7[Online]. Available: http://www.auditec.com
	```

4. Chen, J., Wang, Z., Tuo, D., Wu, Z., Kang, S., & Meng, H. (2022). FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement. arXiv preprint arXiv:2203.12188.
	
	```
	We trained and evaluated FullSubNet+ on a subset of the Interspeech 2021 DNS Challenge dataset. The clean speech set includes 562.72hours of clips from 2150 speakers. The noise dataset includes 181 hours of 60000 clips from 150 classes. During model training, we use dynamic mixing [12] to simulate speech-noise mixture as noisy speech. Specifically, before the start of each training epoch, 75% of the clean speeches are mixed with the randomly selected room impulse response (RIR) from openSLR26 and openSLR28 [27] datasets. After that, the speech-noise mixtures are dynamically generated by mixing clean speeches and noise at a random SNR between -5 and 20 dB. The DNS Challenge provides a publicly available test dataset consisting of two categories of synthetic clips, namely without and with reverberations. Each category has 150 noise clips with a SNR distributed between 0 dB to 20 dB. We use this test set to evaluate the effectiveness of the model.

		- Refernece
		[12] Xiang Hao, Xiangdong Su, Radu Horaud, and Xiaofei Li, “Fullsubnet: A full-band and sub-band fusion model for realtime single-channel speech enhancement,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 6633–6637.
		[27] Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael LSeltzer, and Sanjeev Khudanpur, “A study on data augmentation of reverberant speech for robust speech recognition,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017, pp. 5220–5224.
	```

5. Welker, S., Richter, J., & Gerkmann, T. (2022). Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain. arXiv preprint arXiv:2203.17004.

	```
	We use the standardized VoiceBank-DEMAND [27] dataset for training and testing, as was done in the baseline work (DiffuSE, [10]). We normalize the pairs of clean and noisy audio (x0, y) by the maximum absolute value of x0. We then convert each input into the complex-valued one-sided STFT representation, using F = 512 frequency bins, a hop length of 128 (i.e.,75% overlap) and a periodic Hann window. We randomly crop each spectrogram to 256 STFT time frames during each epoch.

		- Reference
		[10] Y. Lu, Y. Tsao, and S. Watanabe, “A study on speech enhancement based on diffusion probabilistic model,” in Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2021, pp. 659–666.
		[27] C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi, “Investigating RNN-based speech enhancement methods for noiserobust Text-to-Speech.” in SSW, 2016, pp. 146–152.
	```

6. LeBlanc, Ryan, and Sid Ahmed Selouani. "A two-stage deep neuroevolutionary technique for self-adaptive speech enhancement." IEEE Access (2022)

	```
	**Dataset**
	
	Several experiments were carried out to evaluate the proposed methods. The experiments were conducted using 13 sen- tences from the NOIZEUS noisy speech corpus [1]. The corpus uses IEEE sentences downsampled from 25 kHz to 8 kHz [1], [41]. Two real-world nonstationary noise sources were used from the AURORA database, namely, babble and street noise [42]. Four real-world colored noise sources were used. Factory, white and pink noise recordings were taken from the NOISEX-92 noise database [43], and car noise was taken from the AURORA database [42]. Noise was introduced to the clean speech files at −5 dB, 0 dB, 5 dB and 10 dB overall SNR.

		- Reference
		[1] Y. Hu and P. C. Loizou, ‘‘Subjective comparison and evaluation of speech enhancement algorithms,’’ Speech commun., vol. 49, nos. 7–8, pp. 588–601, Jul. 2007.
		[41] E. Rothauser, ‘‘IEEE recommended practice for speech quality measurements,’’ IEEE Trans. Audio Electroacoustics, vol. 17, pp. 225–246, 1969.
		[42] D. Pearce and H.-G. Hirsch, ‘‘The aurora experimental framework for the performance evaluation of speech recognition systems under noisy conditions,’’ in Proc. ISCA, 2000, pp. 29–32.
		[43] A. Varga and H. J. M. Steeneken, ‘‘Assessment for automatic speech recognition: II. NOISEX-92: A database and an experiment to study the effect of additive noise on speech recognition systems,’’ Speech Commun., vol. 12, no. 3, pp. 247–251, Jul. 1993.

	**Metric**
	
	Objective measures were used to evaluate the performance of the speech enhancement methods. The measures were used to evaluate the enhanced speech quality and intelligibility with respect to the corresponding clean speech. The objective measures that were used included:
	
	• The frequency-weighted segmental signal-to-noise ratio (fwSegSNR) [44] was used for objective quality and intelligibility evaluation.
	• The log-likelihood ratio (LLR) [45] was used for objective quality evaluation.
	• The perceptual evaluation of speech quality (PESQ) [46], [47] was used for objective quality and intelligibility evaluation.
	• The normalized-covariance measure (NCM) [48] was used for objective intelligibility evaluation.
	• Three objective composite measures were also used to evaluate the perception of speech [23]. These measures evaluate the predicted rating of signal distortion (SIG), the predicted rating of background noise distortion (BAK), and the predicted rating of overall quality (OVL). Each of the three composite measures uses a five-point scale (1-5).

	The fwSegSNR, LLR, and PESQ were chosen for their high correlation with the overall quality of the signal and signal distortion [23]. The fwSegSNR, NCM, and PESQ were chosen for their good performance in pre- dicting speech intelligibility [48]. fwSegSNR and LLR were mainly included since they were used as optimiza- tion targets in the proposed speech enhancement meth- ods. The composite measures SIG, BAK, and OVL were used to better observe individual aspects of the enhanced speech.
	
	Average objective scores were obtained over the tested speech files under the same noise conditions. Apart from the LLR, which is best minimized, all objective measures were to be maximized for the best results. To further evaluate the speech enhancement performance, a visual representation of the enhanced speech spectrogram was compared to the original and clean spectrograms.

		- Reference
		[23] Y. Hu and P. C. Loizou, ‘‘Evaluation of objective quality measures for speech enhancement,’’ IEEE Trans. Audio, Speech, Language Process., vol. 16, no. 1, pp. 229–238, 2008.
		[44] J. Tribolet, P. Noll, B. McDermott, and R. Crochiere, ‘‘A study of complexity and quality of speech waveform coders,’’ in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., vol. 3, 1978, pp. 586–590.
		[45] S. Quackenbush, T. Barnwell, and M. Clements, Objective Measures of Speech Quality, 1st ed. Eaglewood Cliffs, NJ, USA: Prentice-Hall, 1988.
		[46] Perceptual Evaluation of Speech Quality (PESQ), and Objective Method for End-to-End Speech Quality Assessment of Narrowband Telephone Networks and Speech Codecs, document ITU-T Recommendation P. 862, 2000.
		[47] Mapping Function for Transforming P.862 Raw Result Scores to MOS- LQO, document ITU-T Recommendation P.862.1, 2003.
		[48] M. Jianfen, H. Yi, and P. C. Loizou, ‘‘Objective measures for predicting speech intelligibility in noisy conditions based on new band-importance functions,’’ J. Acoust. Soc. Amer., vol. 125, no. 5, pp. 3387–3405, 2009.
	```

7. Hoang, P., De Haan, J. M., Tan, Z. H., & Jensen, J. (2022). Multichannel Speech Enhancement with Own Voice-Based Interfering Speech Suppression for Hearing Assistive Devices. IEEE/ACM Transactions on Audio, Speech, and Language Processing.
	
	```
	**Dataset**
	
	Speech and Noise Databases: Speech signals used for the own voice, target, and interference, are obtained from the TIMIT database [49]. Speech pauses are removed with an energy-based VAD to minimize the influence of speech gaps in the evaluation. We do not simulate speech gaps caused by conversation pauses. However, the acoustic scene still include situations where neither the own voice nor the target speech are present in a TF tile due to speech being sparse in the TF domain. Hence, there are TF-tiles where own voice or target speech is absent even if they are detected present. The noise database used in the simulation is recordings of noise found in realistic acoustic environments (e.g. a busy canteen and car cabin). The recordings of the noise are made with a spherical microphone array to accurately capture the noise field as measured at a reference point of the spherical microphone array. The captured noise is then transformed and convolved with the AIRs, such that the resulting noise field at the HA microphones in the simulation is identical to the one measured with the spherical microphone array [50].
	
		- Reference
		[49] J. Garofolo et al. “TIMIT: Acoustic-phonetic continuous speech corpus LDC93S1. Web Download,” Philadelphia, PA, USA, Linguistic Data Consortium, 1993.
		[50] P. Minnaar, S. F. Albeck, C. Simonsen, B. Søndersted, S. Oakley, and J. Bennedbæk, “Reproducing real-life listening situations in the laboratory for testing hearing aids,” J. Audio Eng. Soc., 2013.
	```

8. Qian, Y., & Zhou, Z. (2022). Optimizing Data Usage for Low-Resource Speech Recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing.
	
	```
	**Dataset**
	
	The CommonVoice Dataset1 [2] is a massively multilingual corpus of transcribed speech. The contents of the corpus are mainly from Wikipedia articles. We utilize five languages in our experiments, including French (fr), Italian (it), Basque (eu), Por- tuguese (pt), and Catalan (ca). For the traditional approach, these five languages are pooled together for multilingual pretraining, and then the target language is used in finetuning. Table II shows details of languages in CommonVoice. The total speech duration of datasets ranges from 48 hours to 554 hours. We use the June 2020 (v5.1) release of CommonVoice. Note that only part of the dataset is officially validated. 
	
		- Reference
		[2] R. Ardila et al., “Common voice: A massively-multilingual speech cor- pus,” in Proc. 12th Lang. Resour. Eval. Conf., 2020, pp. 4218–4222.
	```

9. Fujimura, T., Koizumi, Y., Yatabe, K., & Miyazaki, R. (2021, August). Noisy-target training: A training strategy for dnn-based speech enhancement without clean speech. In 2021 29th European Signal Processing Conference (EUSIPCO) (pp. 436-440). IEEE.
	
	```
	**Dataset**
	
	
	Table I shows datasets used in experiments. We utilized the VoiceBank-DEMAND [5] which is openly avail- able and frequently used in the literature of DNN-based speech enhancement [3], [4]. The train and test sets consists of 28 and 2 speakers (11572 and 824 utterances), respectively. In addition to this dataset, to evaluate the performance under a training/testing data mismatched condition, we constructed a test dataset by mixing TIMIT [10] (speech) and TAU Urban Acoustic Scenes 2019 Mobile [11] (noise) as TIMIT-MOBILE at signal-to-noise ratio (SNR) randomly selected from −5, 0, 5, and 10 dB. The test sets consist of 1680 utterances spoken by 168 speakers (112 males and 56 females). 
	
	To mimic the use of noisy signals for training in NyTT, we additionally used Libri-Task1 and CHiME5 as noisy datasets. Libri-Task1 consists of mixed signals of the development sets of LibriTTS [14] and TAU Urban Acoustic Scenes 2020 Mobile [15] (TAU-2020) whose SNR was randomly selected from 0, 5, 10, and 15 dB. This dataset includes 8.97 hours of noisy speech with 5736 utterances. CHiME5 was the training dataset of the 5th CHiME Speech Separation and Recognition Challenge [16], and consisted of 77.24 hours of noisy speech with 79967 utterances which was created by cutting each speech interval in the continuous training data with before/after 0.5 sec margin. In addition, we used background noise of CHiME3 [17] as noise dataset (CHiME3).
	
	**Table I**
	
	|Name    				               |Clean Speech               / Noise                         |
	|:--------------------------------:|:---------------------------------------------------------:|
	|VoiceBank-DEMAND[5] + TIMIT-MOBILE| VoiceBank [12], TIMIT [10] /    DEMAND [13], TAU-2019 [11]|
	|Libri-Task1 CHiME5                | Libri-TTS [14] + TAU-2020 [15] Only noisy signal provided |


		- Reference
		[10] John S. Garofolo et al., “TIMIT Acoustic-Phonetic Continuous Speech Corpus LDC93S1,” Web Download. Philadelphia: Linguistic Data Con- sortium, 1993.
		[11] A. Mesaros et al., “Acoustic Scene Classification in DCASE 2019 Challenge: Closed and Open Set Classification and Data Mismatch Setups,” Proc. of DCASE, 2019.
		[12] C. Veaux et al., “The Voice Bank Corpus: Design, Collection and Data Analysis of a Large Regional Accent Speech Database,” 2013 Int. Conf. Orient. COCOSDA held jointly 2013 Conf. Asian Spok. Lang. Res. Eval. (O-COCOSDA/CASLRE), 2013
		[13] J. Thiemann et al., “The Diverse Environments Multi-Channel Acoustic Noise Database: A Database of Multichannel Environmental Noise Recordings,” J. Acoust. Soc. Am., 2013.
		[14] H. Zen et al., “LibriTTS: A Corpus Derived from LibriSpeech for Text- to-Speech,” arXiv:1904.02882, 2019.
		[15] T. Heittola et al., “Acoustic Scene Classification in DCASE 2020 Chal- lenge: Generalization Across Devices and Low Complexity Solutions,” Proc. of DCASE, 2020.
		[16] J. Barker et al., “The Fifth ‘CHiME’ Speech Separation and Recognition Challenge: Dataset, Task and Baselines” Proc. of Interspeech, 2018.
		[17] J. Barker et al., , R. Marxer, E. Vincent, and S. Watanabe, “The Third CHiME Speech Separation and Recognition Challenge: Dataset Task and Baselines”, Proc. of ASRU, 2015.
	```

### 1.2 Papers with paterswith code
- Reference. https://paperswithcode.com/sota/speech-enhancement-on-deep-noise-suppression
	
1. Tzinis, E., Adi, Y., Ithapu, V. K., Xu, B., Smaragdis, P., & Kumar, A. (2022). RemixIT: Continual self-training of speech enhancement models via bootstrapped remixing. arXiv preprint arXiv:2202.08862.		

	```
	**Dataset**
	
	LibriFSD50K (LFSD, in paper, called FSD50K): This data collection includes 45,602 and 3,081 mixtures for training and testing, correspondingly. The clean speech samples are drawn from the LibriSpeech [47] corpus and the noise recordings are taken from FSD50K [48] representing a set of almost 200 classes of background noises after excluding all the human-made sounds from the AudioSet ontology [49]. A detailed recipe of the dataset generation process is presented in [29]. LFSD becomes an ideal candidate for semi-supervised/SSL teacher pre-training on OOD data given its mixture diversity.

	WHAM!: The generation process for this dataset produces 20,000 training noisy-speech pairs and 3,000 test mixtures from the initial WHAM! [50] dataset and has been identical to the procedure followed in [29]. The set of background noises in WHAM! is limited to 10 classes of urban sounds.
	
	VCTK: The VCTK dataset proposed in [51] includes 586 synthetically generated noisy test mixtures, where a speech sample from the VCTK speech corpus [52] is mixed with an isolated noise recording from the DEMAND [53]. The VCTK and DNS test partitions are used to illustrate the effectiveness of RemixIT under a restrictive scenario zero-shot domain adaptation with limited data to perform self-training

		- Reference
		[29] Efthymios Tzinis, Jonah Casebeer, Zhepei Wang, and Paris Smaragdis, “Separate but together: Unsupervised federated learning for speech enhancement from non-iid data,” in Proc. WASPAA, 2021, pp. 46–50.
		[47] Vassil Panayotov,Guoguo Chen,Daniel Povey,and SanjeevKhudanpur, “Librispeech: an asr corpus based on public domain audio books,” in Proc. ICASSP, 2015, pp. 5206–5210.
		[48] Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, and Xavier Serra, “Fsd50k: an open dataset of human-labeled sound events,” arXiv preprint arXiv:2010.00475, 2020.
		[49] Jort F Gemmeke,Daniel PW Ellis,Dylan Freedman,Aren Jansen,Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in Proc. ICASSP, 2017, pp. 776–780.
		[50] Gordon Wichern, Joe Antognini, Michael Flynn, Licheng Richard Zhu, Emmett McQuinn, Dwight Crow, Ethan Manilow, and Jonathan Le Roux, “WHAM!: Extending Speech Separation to Noisy Environments,” in Proc. Interspeech, 2019, pp. 1368–1372.
		[51] Ritwik Giri, Umut Isik, and Arvindh Krishnaswamy, “Attention wave- u-net for speech enhancement,” in Proc. WASPAA, 2019, pp. 249–253.
		[52] Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald, “CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92),” 2019.
		[53] Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent, “The diverse environments multi-channel acoustic noise database (demand): A database of multichannel environmental noise recordings,” in Proc. ICA, 2013.
			
	**Metric**
	
	SI-SDR-WB, PESQ-WB
	```
		
2. Xia, Y., Braun, S., Reddy, C. K., Dubey, H., Cutler, R., & Tashev, I. (2020, May). Weighted speech distortion losses for neural-network-based real-time speech enhancement. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 871-875). IEEE.

	```	
	**Dataset**
	
	We train and evaluate all DNN-based systems using a dataset synthesized from publicly available speech and noise corpus using the MS-SNSD dataset [22] and toolkit ^1. 14 diverse noise types are selected for training, while samples from 9 noise types not included in the training set are used for evaluation. Our test set includes challenging and highly non-stationary noise types such as munching, multi-talker babble, keyboard typing, etc. All audio clips are resampled to 16 kHz. The training set consists of 84 hours each of clean speech and noise while 18 hours (5500 clips) of noisy speech constitute the evaluation set. All speech clips are level-normalized on a per-utterance basis, while each noise clip is scaled to have one of the five global SNRs from {40,30,20,10,0} dB. During the training of all DNN-based systems described below, we randomly select an excerpt of clean speech and noise, respectively, before mixing them to create the noisy utterance.
	
		- Reference
		[1] https://github.com/microsoft/MS-SNSD
		[22] C. K. Reddy, E. Beyrami, J. Pool, R. Cutler, S. Srinivasan, and J. Gehrke, “A Scalable Noisy Speech Dataset and Online Subjective Test Framework,” in ISCA INTERSPEECH 2019, 2019, pp. 1816–1820.
	```
	
	**Metric**
	
	We performed a comparative study of proposed methods with three baselines based on several objective speech quality and intelligibility measures and subjective tests. Specifically, we include perceptual evaluation of speech quality (PESQ) [28], short-time objective intelligibility (STOI)[29], cepstral distance (CD), and scaleinvariant signal-to-distortion ratio (SI-SDR) [30] for objective evaluation of enhanced speech in time, spectral, and cepstral domains. We conducted a subjective listening test using a web-based subjective framework presented in [22]. Each clip is rated with a discrete rating between 1 (very poor speech quality) and 5 (excellent speech quality) by 20 crowd-sourced listeners. Training and qualification are ensured before presenting test clips to these listeners. The mean of all 20 ratings is the mean opinion score (MOS) for that clip. We also removed obvious spammers who consistently selected the same rating throughout the MOS test. Our subjective test complements the other objective assessments, thus providing a balanced benchmark for evaluation of studied noise reduction algorithms.
	
		- Reference
		[22] C. K. Reddy, E. Beyrami, J. Pool, R. Cutler, S. Srinivasan, and J. Gehrke, “A Scalable Noisy Speech Dataset and Online Subjective Test Framework,” in ISCA INTERSPEECH 2019, 2019, pp. 1816–1820.
		[28] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra, “Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs,” in 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No. 01CH37221), 2001, vol. 2, pp. 749–752.
		[29] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “A short-time objective intelligibility measure for time-frequency weighted noisy speech,” in IEEE International Conference on Acoustics, Speech and Signal Processing, 2010, pp. 4214–4217.
		[30] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, “SDR–half-baked or well done?,” in IEEE ICASSP, 2019, pp. 626–630.

### 1.3 Datasets

1. WSJ0 SI-84 dataset (WSJ)
	```
	This dataset consists of 7138 utterances from 83 speakers (42 males and 41 females). Seventy seven speakers are used for training and remaining six are used for evaluation.

		- Reference
		D. B. Paul and J. M. Baker, “The design for the wall street journal-based CSR corpus,” in Proc. Workshop Speech Nat. Lang., 1992, pp. 
	```

2. Non-speech sounds from a sound effect library 

	```
	10000 non-speech sounds from a sound effect library [1, 2]

		- Reference
		[1] www.sound-ideas.com		
		[2] J. Chen, Y. Wang, S. E. Yoho, D. L. Wang, and E. W. Healy, “Large-scale training to increase speech intelligibility for hearing-impaired listeners in novel noises,” J. Acoust. Soc. Amer., vol. 139, pp. 2604–2612, 2016.
	```

3. Babble and cafeteria noises from an Auditec CD
	
	```
	This dataset is for babble and cafeteria noises from an Auditec CD and we generate 150 noisy utterances for both the noises at SNRs of −5 dB, 0 dB, and 5 dB.

		- Reference
		www.auditec.com
	```

4. TIMIT
	
	```
	Speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems
	
		- Reference
		J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, and D. S. Pallett, “DARPA TIMIT acoustic-phonetic continous speech corpus CD-ROM. NIST speech disc 1-1.1,” NASA STIN, vol. 93, 1993, Art no. 27403.
	```

5. NOISEX-92 
	
	```
	150 randomly selected utterances from the 77 training speakers to create a validation set with a babble noise
	
		- Reference
		A. Varga and H. J. Steeneken, “Assessment for automatic speech recognition: II. NOISEX-92: A database and an experiment to study the effect of additive noise on speech recognition systems,” Speech Commun., vol. 12, no. 3, pp. 247–251, 1993.
	```

6. DEMAND noise set

	```
	Park and traffic noise
	
		- Reference	
		J. Thiemann, N. Ito, and E. Vincent, “The diverse environments multi- channel acoustic noise database (DEMAND): A database of multichannel environmental noise recordings,” in Proc. Int. Congr. Acoust., pp. 1–6, 2013.
	```

7. OpenSLR26 and openSLR28

	```
	Room impulse response
	
		- Reference
		Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael LSeltzer, and Sanjeev Khudanpur, “A study on data augmentation of reverberant speech for robust speech recognition,” in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017, pp. 5220–5224.
	```
		
8. VoiceBank
	
	```
		- Reference
		C. Veaux et al., “The Voice Bank Corpus: Design, Collection and Data Analysis of a Large Regional Accent Speech Database,” 2013 Int. Conf. Orient. COCOSDA held jointly 2013 Conf. Asian Spok. Lang. Res. Eval. (O-COCOSDA/CASLRE), 2013
	```

9. Standardized VoiceBank-DEMAND
	
	```
	VoiceBank + DEMAND(Noise)

		- Reference
		C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi, “Investigating RNN-based speech enhancement methods for noiserobust Text-to-Speech.” in SSW, 2016, pp. 146–152.
	```

10. LibriSpeech corpus

	```
	The clean speech samples

		- Reference
		[47] Vassil Panayotov,Guoguo Chen,Daniel Povey,and SanjeevKhudanpur, “Librispeech: an asr corpus based on public domain audio books,” in Proc. ICASSP, 2015, pp. 5206–5210.
	```
		
11. FSD50K

	```
	The noise recordings
	
		- Reference
		[48] Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, and Xavier Serra, “Fsd50k: an open dataset of human-labeled sound events,” arXiv preprint arXiv:2010.00475, 2020.
	```

12. LibriFSD50K 
	
	```
	This data collection includes 45,602 and 3,081 mixtures for training and testing, correspondingly. The clean speech samples are drawn from the LibriSpeech [47] corpus and the noise recordings are taken from FSD50K [48] representing a set of almost 200 classes of background noises after excluding all the human-made sounds from the AudioSet ontology [49]. A detailed recipe of the dataset generation process is presented in [29]. LFSD becomes an ideal candidate for semi-supervised/SSL teacher pre-training on OOD data given its mixture diversity.
	
		- Reference
		[29] Efthymios Tzinis, Jonah Casebeer, Zhepei Wang, and Paris Smaragdis, “Separate but together: Unsupervised federated learning for speech enhancement from non-iid data,” in Proc. WASPAA, 2021, pp. 46–50.
		[47] Vassil Panayotov,Guoguo Chen,Daniel Povey,and SanjeevKhudanpur, “Librispeech: an asr corpus based on public domain audio books,” in Proc. ICASSP, 2015, pp. 5206–5210.
		[48] Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, and Xavier Serra, “Fsd50k: an open dataset of human-labeled sound events,” arXiv preprint arXiv:2010.00475, 2020.
		[49] Jort F Gemmeke,Daniel PW Ellis,Dylan Freedman,Aren Jansen,Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in Proc. ICASSP, 2017, pp. 776–780.
	```

13. WHAM!

	```	
	The generation process for this dataset produces 20,000 training noisy-speech pairs and 3,000 test mixtures from the initial WHAM! [50] dataset and has been identical to the procedure followed in [29]. The set of background noises in WHAM! is limited to 10 classes of urban sounds.
	
	Wsj0 Hipster Ambient Mixtures의 약자예요. 즉 이 데이터셋은 wsj0-2mix 데이터셋에 whisper ai라는 곳에서 만든 고유한 백그라운드 노이즈와 합친 음성 데이터셋 이예요. 더 정확히 설명하면 노이즈는 2018년 말 샌프란시스코의 다양한 urban location 들에서 수집되었어요. 소음 환경은 레스토랑, 카페, 바 그리고 공원이라고 해요. 이 데이터셋은 아쉽게도 wsj0-2mix 라는 데이터셋을 필요로 하는데 wsj0 데이터셋은 Pay이기 때문에 무료로 다운받을 수 있는 데이터셋은 노이즈 뿐이예요.

		- Reference
		[29] Efthymios Tzinis, Jonah Casebeer, Zhepei Wang, and Paris Smaragdis, “Separate but together: Unsupervised federated learning for speech enhancement from non-iid data,” in Proc. WASPAA, 2021, pp. 46–50.
		[50] Gordon Wichern, Joe Antognini, Michael Flynn, Licheng Richard Zhu, Emmett McQuinn, Dwight Crow, Ethan Manilow, and Jonathan Le Roux, “WHAM!: Extending Speech Separation to Noisy Environments,” in Proc. Interspeech, 2019, pp. 1368–1372.
		- https://chloelab.tistory.com/26
	```

14. VCTK

	```
		- Reference
		[52] Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald, “CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92),” 2019.
	```

15. VCTK mixture

	```	
	The VCTK dataset proposed in [51] includes 586 synthetically generated noisy test mixtures, where a speech sample from the VCTK speech corpus is mixed with an isolated noise recording from the DEMAND.

		- Reference
		[51] Ritwik Giri, Umut Isik, and Arvindh Krishnaswamy, “Attention wave- u-net for speech enhancement,” in Proc. WASPAA, 2019, pp. 249–253.		
	```

16. MS-SNSD

	```
	We train and evaluate all DNN-based systems using a dataset synthesized from publicly available speech and noise corpus using the MS-SNSD dataset [22] and toolkit^1. 14 diverse noise types are selected for training, while samples from 9 noise types not included in the training set are used for evaluation. Our test set includes challenging and highly non-stationary noise types such as munching, multi-talker babble, keyboard typing, etc. All audio clips are resampled to 16 kHz. The training set consists of 84 hours each of clean speech and noise while 18 hours (5500 clips) of noisy speech constitute the evaluation set. All speech clips are level-normalized on a per-utterance basis, while each noise clip is scaled to have one of the five global SNRs from {40,30,20,10,0} dB. During the training of all DNN-based systems described below, we randomly select an excerpt of clean speech and noise, respectively, before mixing them to create the noisy utterance.
		
		- Reference
		^1  https://github.com/microsoft/MS-SNSD
		[22] C. K. Reddy, E. Beyrami, J. Pool, R. Cutler, S. Srinivasan, and J. Gehrke, “A Scalable Noisy Speech Dataset and Online Subjective Test Framework,” in ISCA INTERSPEECH 2019, 2019, pp. 1816–1820.
	```

17. NOIZEUS noisy speech corpus

	```
	The experiments were conducted using 13 sen- tences from the NOIZEUS noisy speech corpus [1]. The corpus uses IEEE sentences downsampled from 25 kHz to 8 kHz [1], [41].
	
		- Reference
		[1] Y. Hu and P. C. Loizou, ‘‘Subjective comparison and evaluation of speech enhancement algorithms,’’ Speech commun., vol. 49, nos. 7–8, pp. 588–601, Jul. 2007.
		[41] E. Rothauser, ‘‘IEEE recommended practice for speech quality measurements,’’ IEEE Trans. Audio Electroacoustics, vol. 17, pp. 225–246, 1969.
	```

18. AURORA database

	```
	Two real-world nonstationary noise sources were used from the AURORA database, namely, babble and street noise [42]. Four real-world colored noise sources were used. Factory, white and pink noise recordings were taken from the NOISEX-92 noise database [43], and car noise was taken from the AURORA database [42]. Noise was introduced to the clean speech files at −5 dB, 0 dB, 5 dB and 10 dB overall SNR.

		- Reference
		[42] D. Pearce and H.-G. Hirsch, ‘‘The aurora experimental framework for the performance evaluation of speech recognition systems under noisy conditions,’’ in Proc. ISCA, 2000, pp. 29–32.
		[43] A. Varga and H. J. M. Steeneken, ‘‘Assessment for automatic speech recognition: II. NOISEX-92: A database and an experiment to study the effect of additive noise on speech recognition systems,’’ Speech Commun., vol. 12, no. 3, pp. 247–251, Jul. 1993.
	```

19. NOISEX-92 noise

	```
	Factory, white and pink noise recordings were taken from the NOISEX-92 noise database [43]
	
		- Reference
		[43] A. Varga and H. J. M. Steeneken, ‘‘Assessment for automatic speech recognition: II. NOISEX-92: A database and an experiment to study the effect of additive noise on speech recognition systems,’’ Speech Commun., vol. 12, no. 3, pp. 247–251, Jul. 1993.
	```

20. CommonVoice Dataset1
	
	```
	The CommonVoice Dataset1 [2] is a massively multilingual corpus of transcribed speech. The contents of the corpus are mainly from Wikipedia articles. We utilize five languages in our experiments, including French (fr), Italian (it), Basque (eu), Por- tuguese (pt), and Catalan (ca). For the traditional approach, these five languages are pooled together for multilingual pretraining, and then the target language is used in finetuning. Table II shows details of languages in CommonVoice. The total speech duration of datasets ranges from 48 hours to 554 hours. We use the June 2020 (v5.1) release of CommonVoice. Note that only part of the dataset is officially validated.
	
	**Table II**
	
	|Language    |       #Spk|       #Utt|   Duration|
	|:----------:|----------:|----------:|----------:|
	|Catalan     |      4,742|    317,693|     488 hr|
	|Basque      |        834|     61,426|      88 hr|
	|Portuguese  |        717|     39,072|      48 hr|
	|Italian     |      4,976|     83,407|     130 hr|
	|French      |     11,381|    412,332|     554 hr|
	
		- Reference
		[2] R. Ardila et al., “Common voice: A massively-multilingual speech corpus,” in Proc. 12th Lang. Resour. Eval. Conf., 2020, pp. 4218–4222.
	```

21. Libri-TTS
	
	```
	Speech?
	
		- Reference
		[14] H. Zen et al., “LibriTTS: A Corpus Derived from LibriSpeech for Text- to-Speech,” arXiv:1904.02882, 2019.
	```

22. TAU-2019
	
	```
	Noise
	
		- Reference
		[11] A. Mesaros et al., “Acoustic Scene Classification in DCASE 2019 Challenge: Closed and Open Set Classification and Data Mismatch Setups,” Proc. of DCASE, 2019.
	```

23. TAU-2020
	
	```
	Noise
	
		- Reference
		[15] T. Heittola et al., “Acoustic Scene Classification in DCASE 2020 Challenge: Generalization Across Devices and Low Complexity Solutions,” Proc. of DCASE, 2020.
	```

24. MUSAN
	
	```
	David Snyder, Guoguo Chen, and Daniel Povey, “MU- SAN: A music, speech, and noise corpus,” arXiv preprint arXiv:1510.08484, 2015.
	```

25. Youtube-SoundEffect
	
	```
	https://www.youtube.com/audiolibrary/soundeffects
	```

- [Site closed] Interspeech 2021 DNS Challenge dataset

- Dynamic mixing: Xiang Hao, Xiangdong Su, Radu Horaud, and Xiaofei Li, “Fullsubnet: A full-band and sub-band fusion model for realtime single-channel speech enhancement,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 6633–6637.


26. [TODO][SEARCH][Include below Category] the Device And Produced Speech (DAPS) dataset (Mysore, 2014)
	Mysore, G. J. Can we automatically transform speech recorded on common consumer devices in real-world en- vironments into professional production quality speech?a dataset, insights, and challenges. IEEE Signal Processing Letters, 22(8):1006–1010, 2014.


### 1.4 Metrics
- This evaluation metric implments to https://github.com/ooshyun/Speech-evaluation-methods

1. Perceptual Evaluation of Speech Quality (PESQ) [1]
2. Short-Time Objective Intelligibility (STOI)[2]
3. Cepstral Distance (CD)
4. Scale Invariant Signal-to-Distortion Ratio (SI-SDR) [3] for objective evaluation of enhanced speech in time, spectral, and cepstral domains. 
5. Subjective listening test using a web-based subjective framework presented in [4]
	```
		- Reference
		[1] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra, “Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs,” in 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No. 01CH37221), 2001, vol. 2, pp. 749–752.
		[2] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “A short-time objective intelligibility measure for time-frequency weighted noisy speech,” in IEEE International Conference on Acoustics, Speech and Signal Processing, 2010, pp. 4214–4217.
		[3] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, “SDR–half-baked or well done?,” in IEEE ICASSP, 2019, pp. 626–630.
		[4] C. K. Reddy, E. Beyrami, J. Pool, R. Cutler, S. Srinivasan, and J. Gehrke, “A Scalable Noisy Speech Dataset and Online Subjective Test Framework,” in ISCA INTERSPEECH 2019, 2019, pp. 1816–1820.
	```

## 2. Data

### 2.1 Property
1. Name
2. Numbering on dataset
3. Type
4. Sampling Rate
5. Data Format
6. Numbers of each feature, which (ex. [1]) means the number of 1.3 Datasets

### 2.2 How to download in command
- wget url
- wget -c url: continue download
- wget -r url: download all files in the folder
- wget -q url: progress bar

### 2.3 Dataset Category
1. Utterance
	In spoken language analysis, an utterance is the smallest unit of speech.
	- [1][Pay] WSJ0 SI-84 dataset (WSJ): 7138 utterances from 83 speakers (42 males and 41 females)
			
2. Mix Utterance and Noise
	- [5] NOISEX-92: 150 utterances from the 77 speakers with a babble noise
	
3. Corpus
	In linguistics, a corpus or text corpus is a language resource consisting of a large and structured set of texts.
	- [10][무료] LibriSpeech corpus: https://www.openslr.org/12
		
4. Multilingual Corpus
	- [20] CommonVoice Dataset1

5. Mix Corpus and Noise

6. Speech
	- [4] TIMIT
	- [8] VoiceBank
	- [14][Free] VCTK: https://www.tensorflow.org/datasets/catalog/vctk
	- [21] Libri-TTS: Speech for Text-to-Speech

7. Mix Speech and Noise
	- [9][Free] Standardized VoiceBank-DEMAND: VoiceBank + DEMAND : https://datashare.ed.ac.uk/handle/10283/2791
	- [12] LibriFSD50K: LibriSpeech + FSD50K
	- [13][Pay, Noise dataset is Free] WHAM!: https://wham.whisper.ai
	- [15] VCTK mixture: VCTK + DEMAND
	- [16] MS-SNSD
	- [17] NOIZEUS
	
8. Non-speech sound
	- Noise
		- [3] Auditec CD: Babble and cafeteria noises
		- [6][무료] DEMAND: Park and traffic noise, https://zenodo.org/record/1227121
		- [11] FSD50K: A set of almost 200 classes of background noises after excluding all the human-made sounds from the AudioSet ontology
		- [18] AURORA: Real-world nonstationary noise sources, namely babble and street noise
		- [19] NOISEX-92: Factory, white and pink noise recordings
		- [22] TAU-2019
		- [23] TAU-2020
		- [24] MUSAN
					
	- Sound effect
		- [2] Sound effect library: 10000 non-speech sounds
		- [25] Youtube-SoundEffect
		 	
	- Room impulse response
		- [7] OpenSLR26 and openSLR28: Room impulse response

- MUSDB18: https://sigsep.github.io/datasets/

## 3. Reference
- Dataset: https://wiki.inria.fr/rosp/Datasets
- chime6, 2021: https://chimechallenge.github.io/chime6/track2_data.html
- chime3 + visual: https://cogbid.github.io/chime3av/#download
	
## 4. TODO TO READ

- CHiME2 WSJ0, [https://catalog.ldc.upenn.edu/LDC2017S10]
	```
	CHiME2 WSJ0 was developed as part of The 2nd CHiME Speech Separation and Recognition Challenge and contains approximately 166 hours of English speech from a noisy living room environment. The CHiME Challenges focus on distant-microphone automatic speech recognition (ASR) in real-world environments.
	```

- Tutorials Resources
	- [Speech Separation, Hung-yi Lee, 2020] [[Video (Subtitle)]](https://www.bilibili.com/video/BV1Cf4y1y7FN?from=search&seid=17392360823608929388) [[Video]](https://www.youtube.com/watch?v=tovg5ZxNgIo&t=8s) [[Slide]](http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/SP%20(v3).pdf)

	- [Advances in End-to-End Neural Source Separation, Yi Luo, 2020] [[Video (BiliBili)]](https://www.bilibili.com/video/BV11T4y1774e) [[Video]](https://www.shenlanxueyuan.com/open/course/62/lesson/57/liveToVideoPreview) [[Slide]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/Advances_in_end-to-end_neural_source_separation.pdf)

	- [Audio Source Separation and Speech Enhancement, Emmanuel Vincent, 2018] [[Book]](https://github.com/gemengtju/Tutorial_Separation/tree/master/book)

	- [Audio Source Separation, Shoji Makino, 2018] [[Book]](https://github.com/gemengtju/Tutorial_Separation/tree/master/book)

	- [Overview Papers] [[Paper (Daniel Michelsanti)]](https://arxiv.org/pdf/2008.09586.pdf) [[Paper (DeLiang Wang)]](https://arxiv.org/ftp/arxiv/papers/1708/1708.07524.pdf) [[Paper (Bo Xu)]](http://www.aas.net.cn/article/zdhxb/2019/2/234) [[Paper (Zafar Rafii)]](https://arxiv.org/pdf/1804.08300.pdf) [[Paper (Sharon Gannot)]](https://hal.inria.fr/hal-01414179v2/document)

	- [Overview Slides] [[Slide (DeLiang Wang)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/DeLiangWang_ASRU19.pdf) [[Slide (Haizhou Li)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/HaizhouLi_CCF.pdf) [[Slide (Meng Ge)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/overview-GM.pdf)

	- [Hand Book] [[Ongoing]](https://www.overleaf.com/read/vhdjwcpyryzr)

- Datasets Links
	- Reference. https://github.com/gemengtju/Tutorial_Separation
	- [Dataset Intruduciton] [[Pure Speech Dataset Slide (Meng Ge)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/Speech-Separation-Dataset-GM.pdf) [[Audio-Visual Dataset Slide (Zexu Pan)]](https://github.com/gemengtju/Tutorial_Separation/blob/master/slides/AVSS_Datasets_PanZexu.pdf)

	- [WSJ0] [[Dataset]](https://catalog.ldc.upenn.edu/LDC93S6A)

	- [WSJ0-2mix] [[Script]](https://github.com/gemengtju/Tutorial_Separation/tree/master/generation/wsj0-2mix)

	- [WSJ0-2mix-extr] [[Script]](https://github.com/xuchenglin28/speaker_extraction)

	- [WHAM & WHAMR] [[Paper (WHAM)]](https://arxiv.org/pdf/1907.01160.pdf) [[Paper (WHAMR)]](https://arxiv.org/pdf/1910.10279.pdf) [[Dataset]](http://wham.whisper.ai/)

	- [LibriMix] [[Paper]](https://arxiv.org/pdf/2005.11262.pdf) [[Script]](https://github.com/JorisCos/LibriMix)

	- [LibriCSS] [[Paper]](https://arxiv.org/pdf/2001.11482.pdf) [[Script]](https://github.com/chenzhuo1011/libri_css)

	- [SparseLibriMix] [[Script]](https://github.com/popcornell/SparseLibriMix)

	- [VCTK-2Mix] [[Script]](https://github.com/JorisCos/VCTK-2Mix)

	- [CHIME5 & CHIME6 Challenge] [[Dataset]](https://chimechallenge.github.io/chime6/)

	- [AudioSet] [[Dataset]](https://research.google.com/audioset/download.html)

	- [Microsoft DNS Challenge] [[Dataset]](https://github.com/microsoft/DNS-Challenge)

	- [AVSpeech] [[Dataset]](https://looking-to-listen.github.io/avspeech/download.html)

	- [LRW] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

	- [LRS2] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)

	- [LRS3] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) [[Script]](https://github.com/JusperLee/LRS3-For-Speech-Separationhttps://github.com/JusperLee/LRS3-For-Speech-Separation)

	- [VoxCeleb] [[Dataset]](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

- English

	- [TODO][AMI Corpus](http://groups.inf.ed.ac.uk/ami/corpus/)  
		"The AMI Meeting Corpus is a multi-modal data set consisting of 100 hours of meeting recordings."
		AMI recordings containing overlapping speech, and are evaluated subjectively by human listeners.

	- [Buckeye](http://buckeyecorpus.osu.edu/)

- Different Languages
	- [Voxforge](http://www.voxforge.org)  

	- [SIWIS](http://www.unige.ch/lettres/linguistique/research/current-projects/latl/siwis/)  
	  German, French, English, Italian   
	  40 speakers, -170 utterances each

- Misc Audio
	- [AudioSet](https://research.google.com/audioset/index.html)

- Divide and Remaster (DnR) dataset
	DnR is built from three well-established audio datasets (Lib- riSpeech, FMA, FSD50k), taking care to reproduce conditions simi- lar to professionally produced content in terms of source overlap and relative loudness, and made available at CD quality.


## 5. Dataset downloaded

### 5.1 VoiceBankDEMAND(VoiceBank = VCTK)
- Difference betw/ 56spk vs 28spk 
	- Speaker means person, 
		```
		28 spk = 14 male and 14 female
		56 spk = 28 male and 28 female
		```

- Train and validation
	- Noise database
		- 2 artificially generated, speech shaped noise and babble
			- speech shape: filtering white noise with a filter whose frequency re- sponse matched that of the long term speech level of a male speaker.
			- babble: adding speech from six speakers from the Voice Bank corpus that were not used ei- ther for either training or testing.

		- 8 real-noise recording from Demand database
			- first channel of the 48 kHz
			- domestic noise (inside a kitchen)
			- an office noise (in a meeting room)
			- three public space noises (cafeteria, restaurant, subway station)
			- two transportation noises (car and metro) 
			- a street noise (busy traffic intersection)
		- SNR: 15, 10, 5, 0 dB 
		- Total case: 40 different noisy conditions ( 10 noises x 4SNR)
	- How to add Noise: the ITU-T P.56 method [1] to calculate active speech levels using the code provided in [2], the code of [1] is in P. C. Loizou, Speech Enhancement: Theory and Practice, 1st ed. Boca Raton, FL, USA: CRC Press, Inc., 2007.
		
		```
		[1]: Objective measurement of active speech level ITU-T recommen- dation P.56, ITU Recommendation ITU-T, Geneva, Switzerland, 1993.
		[2]: J.Yamagishi, C.Veaux, S.King, and S.Renals,“Speech synthesis technologies for individuals with vocal disabilities: Voice banking and reconstruction,” J. of Acoust. Science and Tech., vol. 33, no. 1, pp. 1–5, 2012.
		```

	- The clean waveforms were added to noise after they had been normalised and silence segments longer than 200 ms had been trimmed off from the beginning and end of each sentence.

- Test
	- 2 other spk from England of the same corpus, a male and a femail
	- 5 noises from Demand database
		- a domestic noise (living room)
		- an office noise (office space)
		- one transport (bus)
		- two street noises (open area cafeteria and a public square)
	- SNR: 17.5/ 12.5/ 7.5/ 2.5 dB
	- Total case: 20 different noisy conditions (5 noises x 4 SNRs)

### 5.2 DEMAND Noise set, https://zenodo.org/record/1227121
	```
	The DEMAND (Diverse Environments Multichannel Acoustic Noise Database) presented here provides a set of recordings that allow testing of algorithms using real-world noise in a variety of settings. This version provides 15 recordings. All recordings are made with a 16-channel array, with the smallest distance between microphones being 5 cm and the largest being 21.8 cm.

	Recording equipmentThe array uses 16 Sony ECM-C10 omnidirectional electret condenser microphones. They are connectedto a Inrevium / Tokyo Electron Device TD-BD-16ADUSB USB D/A converter.  The converter wasconnected to laptops running either Microsoft Windows or the Linux operating system; the choice ofoperating system should not have affected the recordings.The data was captured using the tools supplied with the USB converter and stored in its customformat1. The MATLAB scriptich2wav.mprovided on the website was then used to trim the data andconvert it to the standard RIFF (“.wav”) format.Each environment noise recording is available as a set of 16 individual mono sound files in a subdi-rectory (e.g.DKITCHEN/ch01.wav), packaged in a “zip” file. The resampling from 48 kHz to 16 kHz was done using the standard “resample()” function in MATLAB R2012a.

	DC offset
	The DC offset of each channel can be found by a simple average ofthe PCM data within each channel.We found that the DC offset in all channels is less than the A/D converter step size.4.2  
	
	
	Gain variations
	The recorded signals were not subject to any gain normalization. Therefore, the original noise power ineach environment is preserved.
	Given the size of the microphone array compared to the distances of the noise sources in each en-vironment, we expect that the overall level of sound at each microphone should be roughly the same,barring occlusion effects from the support structure. However, the microphones of the array are electretmicrophones, and contain internal preamplifiers. They are not calibrated with respect to each other, andso gain variations are to be expected: we found that the energy in some channels is consistently higherthan in other channels. Algorithms working on this data should compensate for this variation
	```

### 5.3 Others
- DNS Challenge 2022
- [Speech] VCTK
- [OnlyNoise]  WHAM
- [Music] MUSDB18
- [Corpus] LibriSpeechASRcorpus
- [Noise, Anotated Speech, Speech, Music, Noisy] Clarity Challenge