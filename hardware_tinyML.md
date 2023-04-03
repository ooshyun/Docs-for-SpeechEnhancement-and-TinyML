# Hardware Platform/Framework/Library for Tiny ML

@updated date 2022.12.31

## 1. Ongoing Target
### 1.1 TinyLSTMs: Efficient Neural Speech Enhancement for Hearing Aids
	
- Target MCU: STM32F746VE MCU

- Product Specification
	- ARM 32bit Cortex M7 with FPU
	- Flash Memory 1 MB
	- Frequency up to 216 MHz
	- SRAM 320 KB: (including 64 KB of data TCM RAM for critical real-time data) + 16KB of instruction TCM RAM (for critical real-time routines) + 4 KB of backup SRAM (available in the lowest power modes)

- Specification
	- num 		: 
	- data type	: Integer (uint8, int8, int16, int24, int32, float32, float64)
	- Latency ~ Compute Complexity
	- Compute Complexity : 1.55MOPs/inf > 10ms under latency on target MCU
		MOps/inf denotes 10^6 operations per frame inference, which means one inference need Millon Operations Per Second (MOPs) to complete
	- Model Size : < 0.5MB(500KB)
	- Working Memory : < 320KB
	- GPUH: GPU hours

- Result
	- Referring to paper
	- Param: 
	- Compute Complexity: 1.78X 
	- model size: 11.9X
	- Latency: Limit, 12.52 ms -> 4.26 ms
	- GPUH: 255X

### 1.2 Test remaining Memory 
- ‘text’ is your code, and constants (and also the vector table).
- ‘data’ is for initialized variables. This is count towards both RAM and FLASH. The initialized value allocates space in FLASH which then is copied from ROM to RAM in the startup code.
- ‘bss’ is for the uninitialized data in RAM which is initialized with zero in the startup code.
- dec = .text + .data + .bss

### 1.3 Target Specification
- num 		: 
- data type	: Integer (uint8, int8, int16, int24, int32, float32, float64)
- Latency ~ Compute Complexity
- Compute Complexity : 
	- 1.55MOPs/inf -> 10ms under latency on target MCU
	- Expectation x5-x6 less than Paper 1
	- MOps/inf denotes 10^6 operations per frame inference
- Model Size : < 0.5MB 
- Working Memory : < 320KB

	- Most Popular model in Interspeech 2022: ConvRNNT, tacotron and gan
	- 근데 왜 Fixed point를 쓸까? Floating point가 부담되나? FPU가 없는 경우 MIPS가 Floating point를 계산하는데 더 많은 비용이 든다.
	- Measurement: Power, The real-time factor, xRT

## 2. Board list / Kernels based on Tensorflow lite in Tensorflow site
### 2.1 Board list
Reference. https://www.tensorflow.org/lite/microcontrollers
	```
	Arduino Nano 33 BLE Sense
	SparkFun Edge
	STM32F746 Discovery kit
	Adafruit EdgeBadge
	Adafruit TensorFlow Lite for Microcontrollers Kit
	Adafruit Circuit Playground Bluefruit
	Espressif ESP32-DevKitC
	Espressif ESP-EYE
	Wio Terminal: ATSAMD51
	Himax WE-I Plus EVB Endpoint AI Development Board
	Synopsys DesignWare ARC EM Software Development Platform
	Sony Spresense
	```

### 2.2 Board list w/ examples
- Reference. https://github.com/tensorflow/tflite-micro
	```
	Arduino	Arduino Antmicro
	Coral Dev Board Micro	TFLM + EdgeTPU Examples for Coral Dev Board Micro
	Espressif Systems Dev Boards	ESP Dev Boards
	Renesas Boards	TFLM Examples for Renesas Boards
	Silicon Labs Dev Kits	TFLM Examples for Silicon Labs Dev Kits
	Sparkfun Edge	Sparkfun Edge
	Texas Instruments Dev Boards	Texas Instruments Dev Boards
	```

### 2.3 Kernal based on Tensorflow lite github
- What is the Kernel? directly connecting to Hardware and getting a resources
	```
	Cortex-M
	Hexagon
	RISC-V
	Xtensa
	```
1. Xtensa Hifi 
	- In our case
		- Hifi Mini, it is based on Hifi2 library(-2009)
		- Hifi2 case is not on the Cadence page.
		- Refernece
			https://www.cadence.com/en_US/home/training/all-courses/86057.html
			https://ip.cadence.com/knowledgecenter/search-results?mact=Search%2Ccntnt01%2Cdosearch%2C0&cntnt01returnid=756&cntnt01searchinput=hifi+2&submit=Submit&cntnt01searchfilters%5B%5D=Search&cntnt01searchfilters%5B%5D=Uploads&cntnt01searchfilters%5B%5D=News&cntnt01origreturnid=330
			
	- In Candence page, they treat the products as below,
		HiFi 1 DSP, HiFi 3 DSP, HiFi 3z DSP, HiFi 4 DSP, HiFi 5 DSP
		- Reference
			https://www.cadence.com/content/dam/cadence-www/global/en_US/documents/ip/tensilica-processor-ip/hifi-dsps-ds.pdf

	- Framework
		- Reference: https://www.cadence.com/en_US/home/tools/ip/tensilica-ip/hifi-dsps/software.html#frameworks
		```
		XAF (Xtensa Audio Framework)	Run-time framework for creating and managing audio and speech processing pipeline chains	Cadence license
		Sound Open Framework			Audio DSP firmware infrastructure and drivers	BSD/MIT licensed firmware BSD/GPL licensed drivers
		Audio Weaver					Graphical UI for real-time tuning and debugging of audio and voice algorithms	DSP Concepts
		TFLM (Tensor Flow Lite Micro)	Industry-standard ML framework for neural network inferencing on resource-constrained devices	Creative Commons Attribution 4.0 Apache 2.0
		```
		
	- Library
		- Reference: https://www.cadence.com/en_US/home/tools/ip/tensilica-ip/hifi-dsps/software.html#libraries
		```
		NDSP Lib (Nature DSP Library)		Optimized math and signal processing library						Cadence license
		CMSIS Lib (Math and DSP libraries)	Optimized math and signal processing library supporting CMSIS API	Cadence license
		NN Lib (Neural Network Library)		Optimized NN operators used in speech and audio workloads			Cadence license
		Codecs	Audio, speech/voice codecs (note to Web Team: provide link to list)								Cadence license and codec-dependent license
		```
		
	- Operating Systems
		- Reference: https://www.cadence.com/en_US/home/tools/ip/tensilica-ip/partners.html#operatingsystems

	- Currently, in Tensorflow lite github, they tested below products, but failed (2022.04.19).
		```
		Hifi4 / Fusion F1
		Hifi5
		Vision P6	
		```
2. Others
	- Qualcomm case: Kalimba DSP core, Hexagon Kernel
	- Cortex-M: arm nn, CMSIS-DSP
