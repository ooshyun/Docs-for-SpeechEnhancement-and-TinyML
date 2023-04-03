# Cost for GPU Hardware
## TL;DR
1. 서버 vs 하드웨어	
	RTX3090 의 성능으로 하루에 3시반30분 사용한다고 가정한다면 아마존 서비스를 이용하는 것이 300일 기준으로 이상 쓸 경우 더 비싸다.
	또한 많은 트위터 의견에서, 아마존 Instance의 경우 좋지 않은 케이스가 많다고 보고됐다.

2. 하드웨어 업그레이드	
	https://docs.google.com/spreadsheets/d/1g_1zCvu1dXSD41DuCuFMa-PgKh_I-mW2vHmAyB37aP4/edit#gid=2100100521 
	Source Hardware 우측 상단에 보면, 업그레이드 해야하는 것은 GPU, HDD, SDD, RAM, POWER를 업그레이드 해야하며 비용은 190만원으로 예상한다.

- 모델별로 FLOPs 계산 참조
	1. https://github.com/sovrasov/flops-counter.pytorch/issues/16
	2. https://github.com/facebookresearch/SlowFast/blob/0cc82440fee6e51a5807853b583be238bf26a253/slowfast/utils/misc.py#L106
	3. https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md


## GPU Cloud Setup
### Blog. 어떤 클라우드 GPU를 사용하세요?
현재 저는 1개의 GTX 1080을 사용하고 있긴 하지만, 가끔 더 빨리 다양하게 실험해 보고 싶단 생각이 들 때 GPU 인스턴스를 써볼까? 라는 생각이 문득 문득 들더라구요.
A. 살펴보고 있는 것 중에서는
1. [$0.7 ~ 0.77/h] Google Cloud GPU
	- 가격(Reference. https://cloud.google.com/products/calculator#id=)
		Option
		- Instance
			Number
			Operating System/ Software
			Provising model
			Machine Family
			Series
			Machine type(vCPUs, RAM)
			Threads per core
			Boot disk type
			Boot disk size
			GPU
				Datacenter location
				Instances using ephemeral public IP
				Instatnces using static public IP
				Committed usage
				Average hours per data each server is running
				units (hours or minutes)
				per day or per month
				Average days per week each server is running
		- Sole-tenant nodes
			Number of nodes
			Node type(vCPUs, RAM)
			GPU types
				Type NVIDIA Tesla P100, P4, V100(X), T4
				number of GPU 0, 1(X), 2(X), 4 
			CPU Overcommit(T or F)
			Local SSD
			Datacenter location
			Committed usage
			Average hours per data each server is running
			units (hours or minutes)
			per day or per month
			Average days per week each server is running
		- [Not Use]Persistent Disk
	- Details
		https://cloud.google.com/gpu/
		https://cloud.google.com/compute/docs/gpus/
		GPUs are now available for Google Compute Engine and Cloud Machine Learning
		https://cloud.google.com/blog/products/gcp/gpus-are-now-available-for-google-compute-engine-and-cloud-machine-learning?fbclid=IwAR2vkEzpD80K3V2ouoAglckI2wR5XJEJWoZ-jzV9KHSMw73iTDDYRurxLc4
		Running Jupyter notebooks on GPU on Google Cloud
		https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd
		Google Cloud TF-GPU Setup for Ubuntu 16.04
		https://docs.google.com/presentation/d/1_smMnYYjzR_ParHtrilf9hr4cTY3ot8TuhqkKzM-fu8/edit?fbclid=IwAR2bB7RWak8VXB9vi09nchqPgEyowSKA_jBtCLEpPWFInA5nXvGNjJgC_2w#slide=id.p
		cs231n도 이 방식을 사용하네요.
		http://cs231n.github.io/gce-tutorial-gpus
		P100을 지원할 예정이라는데, 아직은 K80인가 봅니다. 동경 리전은 아직 GPU를 지원하지 않고 있구요. 대만 리전은 지원하고 그나마 가깝네요. K80 1개를 쓰는데 시간 당 $0.7 ~ 0.77 정도의 가격( https://cloud.google.com/compute/pricing )입니다. 그런데 튜토리얼에 나와있는데로 꼭 나중에 GPU quota를 늘리는 방식으로 해야하는 것일까요? (아마 처음 $300 어치 트라이얼 상태에서 셋업할 때 비용이 들지 않기 위해서 이렇게 하는 것이 아닐까 추측해 봅니다)
		* 대만 리전이 일본 리전과 역시 차이가 좀 있네요. 아래 링크의 블로그에서 실험해 보신 분 이야기로는 대만 72ms, 일본 47ms 라고 합니다.
		[Not access]https://blog.dreamyoungs.com/.../01/gcp-asia-northeast-test/
2. [$1.465/h] 아마존의 P2, G3 인스턴스
	Running Jupyter notebooks on GPU on AWS: a starter guide
	https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html?fbclid=IwAR0y40i5a1OIJxsFPlOFDZ707vkytjUujdCV_66knDvDs5uJRMYZAlxV-CI
	New – Next-Generation GPU-Powered EC2 Instances (G3)
	https://aws.amazon.com/blogs/aws/new-next-generation-gpu-powered-ec2-instances-g3/?fbclid=IwAR1Jcea5tS4PEa7wWiCIBOHI96Uw7KaKx9LmblxoERUUXJMJNe5_Eqr3Soo
	스팟 인스턴스가 아니라 온디맨드로 쓰면,
	https://aws.amazon.com/ko/ec2/pricing/on-demand/
	현재 서울 리전은 G2, G3 인스턴스가 없고 P2 인스턴스만 있습니다.
	P2는 K80, G3는 M60 NVidia GPU를 사용할 수 있구요.
	서울 리전의 p2.xlarge가 시간당 $1.465로 나옵니다.
3. [$0.9 ~ $1.093/h] Azure
	Azure N-Series
	https://azure.microsoft.com/en-us/blog/azure-n-series-general-availability-on-december-1/?fbclid=IwAR2rMZkCiwI-eof0Mvlf3lOgKA-O8asuizlnzsg--kI4-AwJQGSdda7Nnjs
	Install NVIDIA GPU drivers on N-series VMs running Linux
	https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup?fbclid=IwAR2lMTO9zMSLbSb52nZW71ydLSrkPfyqFXzeItQOxsIV7FbI3Rq9vqg7rjI
	Linux Virtual Machines Pricing
	https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/?fbclid=IwAR0rbM2s6nLIylGSm1BFLng6-TgnYcFnglL-7zDF9HDMENhCYXpnprrfuEk
	을 보면 K80이 있는 NC6가 시간당 $0.9, M60이 있는 NV6가 시간당 $1.093 이네요.
4. IBM
	BlueMix는 가격이 높아서 잘 살펴보지 않았어요.
	https://www.ibm.com/cloud-computing/bluemix/gpu-computing
5. FLOYD
	아니면 DL쪽의 Heroku를 표방하고 있는,
	https://www.floydhub.com
	에서 실험해 보는 것도 가능할 것 같아요. 연속 1시간 정도 돌리는 것은 무료더라구요. K80 정도 사용하는 것 같구요.
6. Nvidia cloud
	https://www.nvidia.com/en-us/data-center/gpu-cloud-computing/

- Details
	```
	레딧도 좀 찾아봤습니다.
	GPU cloud computing prices
	https://www.reddit.com/r/MachineLearning/comments/5md7ud/d_gpu_cloud_computing_prices/
	AWS GPUs vs. your own machine
	https://www.reddit.com/r/MachineLearning/comments/5veyyb/discussion_aws_gpus_vs_your_own_machine/
	전기세 이야기가 많이 나오네요. ㅎㅎ
	B. 그래서 질문은
	1. 개인이 빠른 실험을 목적으로 간헐적으로 강력한 GPU를 쓰기에 좋은 클라우드 GPU는 어떤 것일까요? (또는 요즘 사용하고 계신 GPU 인스턴스가 있다면 무엇이고 어떤 이유에서 선택하셨는지 궁금합니다)
	2. 원격으로 주피터 노트북을 붙여 쓰고, 데이터를 업로드하거나 결과(특히 많은 이미지들)를 다운로드 하려면 핑이 잘 나오는 리전을 써야 할까요?
	3. 앞서 나열한 GPU 인스턴스들은 물리적인 GPU를 고스란히 할당하는 것일까요? 뭔가 나눠 쓰는 것일까요?
	4. 아이패드 같은 태블릿에서 GPU 인스턴스에 접근해서 브라우저로 주피터 노트북을 다루는 것이 가능하고 또 쾌적할까요?
	
	* @hardmaru의 최근 포스팅을 보면 GCE를 사용하네요. GPU가 아니라 64 core의 CPU구요.
	GCE 64 core machines w/ 416GB of RAM are pretty sweet for experimenting with evolution strategies without writing distributed training code.
	https://twitter.com/hardmaru/status/889530242934505472
	구글 브레인 레지던시였어서 이런 것을 잘 아는걸까요... 최근에 이런 내용을 본 것도 기억이 납니다.
	Benchmarking TensorFlow on Cloud CPUs: Cheaper Deep Learning than Cloud GPUs
	http://minimaxir.com/2017/07/cpu-or-gpu/
	```
- Comment 
	정리가 아주 잘되있으신 글 공유해주셔서 감사합니다. 저의 경우에는 Azure 와 AWS를 사용하고 있으며 간헐적으로 쓴다면 Azure는 분당과금으로 분당 쓴만큼 내며 AWS는 시간당 과금으로 1분써도 한시간 입니다. 위치로 생각하면 AWS가 한국에 있으므로 AWS가 좋구요. AWS 는 K80 Azure 는 M60 이므로 정확한 비교는 프로그램에 따라 다를듯합니다.

- Reference
	- 본 글 출처: https://www.facebook.com/groups/TensorFlowKR/posts/509780449363018/
	- GPU 서버 가격 링크 모음: https://github.com/zszazi/Deep-learning-in-cloud
	- 기사, 카카오도 GPU CLOUD?: http://it.chosun.com/site/data/html_dir/2021/11/25/2021112502055.html'

### Local Server, 딥러닝용 서버 견적(보이저엑스 남세동 대표의 페이스북 글 참조)  
```
TF-KR의 여러분 안녕하세요. 항상 많은 도움 받고 있습니다.
딥러닝 스타트업인 저희 회사(보이저엑스)에서는 딥러닝 개발자 개인용 컴퓨터에 GPU 2개씩 꼽고 있습니다. 물론 공용 머신러닝 기계들(DGX A100 Station 등)은 따로 있고요.
아시다시피 컴퓨터 사양 맞추는게 생각보다 시간 잡아먹는 일인데요. 최근에 구매한 개인용 3090 2Way 사양 공유하고 코멘트 드립니다. 고민 시간 아끼는데 도움이 되길 바라며 공유 드립니다.
이게 정답이라는 얘기는 당연히 아니고요. 상황에 따라 정답도 다를테고요. 다만, 이런 답도 있다는 것만으로도 시간 아끼는데 도움이 되실 듯 하여 공유 드리는 것으로 이해해 주시기 바랍니다.
참, 3090 가격이 하늘을 찔렀기에 (그냥 옛날부터 갖고있던 1080Ti, 2080Ti로 버티면서) 구매를 미뤄왔는데요. 요즘에는 3090 가격이 우주로 가고 있고, 언제 안드로메다까지 갈지 몰라서 더 이상 못 미루고 구매했습니다.
------------------------------------------------------------
중요순으로 코멘트:
1. GPU: Gigabyte RTX 3090 Gaming OC D6X 24GB * 2개: 요즘 그래픽 카드 구하기가 너무 힘드므로 그냥 유명/인기 브랜드 3개 (MSI, Gigabyte, ASUS) 중에서 구해지는대로 구한 겁니다. 🙂 슬롯 간격과 발열 생각하면 블로워 타입이 좋은데 블로워 타입은 시장에서 사라지는 중이라 합니다. 이게 다 코인 때문입니다.
2. Mainboard: ASRock Z490 PG Velocita - 특이하게도 PCIe 슬롯 간격이 대단히 넓습니다. 이런 보드가 많지 않습니다. 그래서 매우 두껍고 발열 문제가 많은 RTX 3090 카드를 두개 꼽고 사용하는데 유리할 것으로 기대합니다. 참고로 이 보드는 SLI를 지원하지 않는 것이 아쉽습니다만 SLI가 학습 속도에 주는 영향은 보통 5% 내외 정도라서 그냥 포기했습니다.
3. Power: SuperFlower SF-1600F14HT LEADEX TITANIUM - 1300W로 할까 하다가 1600W로 했습니다. 사실 오버클럭 안하면 1300W로도 괜찮을 듯 하지만 안정성 문제가 생겼을 때 의심 포인트를 하나라도 줄이기 위해서 그냥 1600W로 했습니다.
4. Memory: 삼성전자 DDR4 32G PC4-25600 * 2개 - GPU 다음으로 학습 속도에 영향을 줄 수 있는 부분이라 가급적 빠른 것으로 하되 또 안정성과 가성비는 따져야 하겠기에 그냥 초록색 삼성 메모리로 합니다.
5. SSD: 삼성전자 980 Pro M.2 NVME 1TB - GPU, 그리고 Memory 다음으로 학습 성능에 영향을 줄 수 있는 부분이라 성능 좋고 안정적인 것 중에 가장 일반적인 것, 별 생각없이 고를 수 있는 것으로 고른 것입니다.
6. CPU: i7-10700K - 물론 구체적으로 어떤 일을 하느냐에 따라 다르지만 딥러닝 할 때 보통은 CPU가 그렇게 큰 역할을 하지는 않습니다. 저희도 보통은 그래서 그냥 고사양 중에서 가장 평범한 선택지라고 할 수 있는 것으로 한 것입니다.
7. HDD: Toshiba 4TB X300 HDWE140 (SATA3/7200/128M) - 여기서부터는 크게 중요해 보이지는 않습니다만 아무리 그래도 7200RPM, 128M 정도 되는 것은 골라야 답답하지 않겠습니다. 다나와 인기순 상위권에서 적당히 고른 것입니다.
8. 쿨러: 쿨러마스터 HYPER 212 LED TURBO WHITE - 그냥 다나와에서 평 좋은 것 고른 것입니다.
9. 케이스: ABKO SUITMASTER 603G 아우라 - 그냥 조립업체에서 골라준 것입니다.
------------------------------------------------------------
이렇게 하면 요즘 시세로 대략 딱 1,000 만원 정도 합니다. 원래 개발자 컴은 1080Ti * 2개 해서 전체 500만원 정도에 맞추고 있었는데 작년에는 2080Ti * 2개 해서 전체 700만원 정도로 오르더니 요즘에는 3090 * 2개 해서 전체 1,000만원(2022.05 기준, 750만원) 정도 하게 되었습니다. 이게 다 비트코인 때문입니다.
```

### Hardware Setup, romaglushko, [1]
1. Reason to setup
	- Dependency error because of version mismatch
	- Always plant to experiement with configuration
2. Hardware
	- 용어
		SLI 2-3개 확장
		ATX 메인보드 표준 크기규격
		CPU, K - 오버클럭, F - 내장그래픽 X
	- GPU
		Suppliment: NVIDIA
		The core unit, named Tensor cores operates multiplying 4x4 matrices in 1 operation. The entire operation is as below
		1. RAM in GPU > Global GPU Memory 
			CPU threads load preprocessed batches into entirely separate GPU device memory, not CPU RAM. The device memory is the slowest kind of memory in the GPU.
		2. Global GPU Memory > Shared Memory 
			Shared memory is 10-50x faster than the global GPU memory, but it's also much smaller (normally hundreds of Kbs). This memory is purely available for a Streaming Multiprocessor (SM) that is an analogue of CPU core in GPU architecture. Data is stored there in so-called tiles.
		3. Shared Memory > Tensor Core Registries
			Streaming Multiprocessors operates their tensor cores in parallel and upload part of the tiles into tensor core registries.
		So any bottlenecks in data loading flow would lead to suboptimal utilization of tensor cores, no matter how many of them you have in your GPU.
		[TODO: source -> Check]
		**Main GPU features**
		1. Global GPU Memory 
			It defines how big batch sizes you can use during training or how quality samples you can use if it comes to computer vision
		2. Memory Bandwidth 
			It is a rate at which data is being transferred inside of the device
		3. Architecture
			the more recent architecture is the better. Newer architectures may be better in terms of shared memory size, feature set (like mixed precision computations) and could be more efficient in terms of wattage per effective computations metric.
		4. Performance per Cost
		5. GPU Cooling
		6. GPU Wattage
	- Motherboard
		1. most of the I/O ports (like USB, Ethernet, etc)
		2. chipset with BIOS
		3. WiFi, Bluetooth adapters
		- Checkpoint
			1. Compatable w/ CPU
			2. num PCI ports(more space better cooling)
			3. [TODO Check] Compatable PCI w/ GPU 
			4. WiFi/Bluetooth adapter
	- CPU
		1. Preprocessing Dataset
		2. Loading batches into RAM
		3. Transmitting batches from RAM to the GPU global memory
		4. Running functions in GPU device
		- Checkpoint: num threads and cores
	- RAM
		It's better to have enough RAM to run model training without falling back to swapping. Larger capacity would allow to run bigger batches of data and execute more data loaders to make GPU wait less.
		- That's why some recommend RAM >= 2xGPU Memory 
	- Storage
		- Expect: SSD 1T, HDD 4T
	- Power System Unit
		- Typically there are two the most power consuming components: CPU and GPUs.
		- Adds another 10-15% on top of that (for other components, overclocking, etc)
		- num slots and connectors that the PSU provides.
	- Cooling
		Making sure there is enough space between GPUs is a great no-overhead way to cool your system. It's particularly good in 1-2 GPUs setup. A water cooling is a good option for 3+ GPUs setups.
		- 추천 제조사: NOCTUA, DEEPCOOL
	- PC case
		- Additional system coolers
		Additional I/O ports (USB, Type-C, etc)
		- Cable management
		- Vertical slots for GPUs
		- Slots for SATA disks
		- Cool design and look (well, that may matter)
		- One of the most important considerations is the ability to hold all your cooling systems and GPUs. If your case is too small, it may be problematic.
		- [고려사항] 그래픽카드 -> 쿨링(쓰로틀링)
			- 추천 제조사: 리안리, Phanteks, Fractal Design, be quiet
	- Writer's Choice
		GPU: Gigabyte GeForce RTX 3070 8Gb Aorus Master
		Motherboard: MSI x470 Gaming Plus
		CPU: AMD Ryzen 5 3600
		RAM: G.Skill Ripjaws V Series 32Gb (2 x 16Gb)
		Storage: Samsung 970 Evo 500Gb M.2-2280 NVME
		Cooling: Cooler Master Hyper 212 Black Edition
		PSU: EVGA G2 750W 80+ Gold
		PC Case: NZXT H510
		Wireless Adapter: TP-Link Archer T2U Plus	
	- Difference GPU betw/ ventor
		1. Max Power Limit
			It affects GPU performance.
		2. Quality of the card cooling system 
			the ability to cold card and sustain performance for a long time.
		3. Fan Noise
			It's just annoying to run noisy cards 

### Which GPU(s) to Get for Deep Learning, timedettemers, [2]
1. The Most Important GPU Specs for Deep Learning Processing Speed
	1. Tensor cores
		- Tensor Cores reduce the used cycles needed for calculating multiply and addition operations, 16-fold — in my example, for a 32×32 matrix, from 128 cycles to 8 cycles.
		- Tensor Cores reduce the reliance on repetitive shared memory access, thus saving additional cycles for memory access.
		- Tensor Cores are so fast that computation is no longer a bottleneck. The only bottleneck is getting data to the Tensor Cores.
		- Cycle timings or Latencys for operation
			Here are some important cycle timings or latencies for operations:
				- Global memory access (up to 48GB): :200 cycles
				- Shared memory access (up to 164 kb per Streaming Multiprocessor): :20 cycles
				- Fused multiplication and addition (FFMA): 4 cycles
				- Tensor Core matrix multiply: 1 cycle
			Furthermore, you should know that the smallest units of threads on a GPU is a pack of 32 threads — this is called a **warp**. Warps usually operate in a synchronous pattern — threads within a warp have to wait for each other. All memory operations on the GPU are optimized for warps. For example, loading from global memory happens at a granularity of 32x4 bytes(32 floats), exactly one float for each thread in a warp. We can have up to 32 warps = 1024 threads in a **streaming multiprocessor (SM)**, the GPU-equivalent of a CPU core. The resources of an SM are divided up among all active warps. This means that sometimes we want to run fewer warps to have more registers/shared memory/Tensor Core resources per warp.
				- 1 warp = 32 threads ~ 1 thread = 4 bytes ~ 1 warp = 32 floats ~ 32 warps = 1024 threads in a SM			
		- Comparison w/ Tensor Cores, the condition it A @ B = C, size 32x32
			For both of the following examples, we assume we have the same computational resources. For this small example of a 32×32 matrix multiply, we use 8 SMs (about 10% of an RTX 3090) and 8 warps per SM.
			1. w/o Tensor Cores
				- If we want to do an A @ B=C matrix multiply, where each matrix is of size 32×32, then we want to load memory that we repeatedly access into shared memory because its latency is about ten times lower (200 cycles vs 20 cycles). 
				
				- A memory block in shared memory is often referred to as **a memory tile** or just a tile. Loading two 32×32 floats into a shared memory tile can happen in parallel by using 2x32 warps. 
				We have 8 SMs with 8 warps each, so due to parallelization, we only need to do a single sequential load from global to shared memory, which takes 200 cycles.

				- To do the matrix multiplication, we now need to load a vector of 32 numbers from shared memory A and shared memory B and perform a fused multiply-and-accumulate (FFMA). Then store the outputs in registers C. We divide the work so that each SM does 8x dot products (32×32) to compute 8 outputs of C. Why this is exactly 8 (4 in older algorithms) is very technical. I recommend Scott Gray’s blog post on matrix multiplication to understand this. This means we have 8x shared memory access at the cost of 20 cycles each and 8 FFMA operations (32 in parallel), which cost 4 cycles each. 
				- In total, we thus have a cost of:
					200 cycles (global memory) + 8x20 cycles (shared memory) + 8x4 cycles (FFMA) = 392 cycles
			2. w/  Tensor Cores
				- With Tensor Cores, we can perform a 4×4 matrix multiplication in one cycle. To do that, we first need to get memory into the Tensor Core. Similarly to the above, we need to read from global memory (200 cycles) and store in shared memory. To do a 32×32 matrix multiply, we need to do 8×8=64 Tensor Cores operations. 
				
				- A single SM has 8 Tensor Cores. So with 8 SMs, we have 64 Tensor Cores — just the number that we need! We can transfer the data from shared memory to the Tensor Cores with 1 memory transfers (20 cycles) and then do those 64 parallel Tensor Core operations (1 cycle). 

				- This means the total cost for Tensor Cores matrix multiplication, in this case, is:
					200 cycles (global memory) + 20 cycles (shared memory) + 1 cycle (Tensor Core) = 221 cycles.
		- Conclusion
			Thus we reduce the matrix multiplication cost significantly from 392 cycles to 221 cycles via Tensor Cores. In this simplified case, the Tensor Cores reduced the cost of both shared memory access and FFMA operations. 
			While this example roughly follows the sequence of computational steps for both with and without Tensor Cores, please note that this is a very simplified example. Real cases of matrix multiplication involve much larger shared memory tiles and slightly different computational patterns.
			However, I believe from this example, it is also clear why the next attribute, **memory bandwidth, is so crucial for Tensor-Core-equipped GPUs**. Since global memory is the most considerable portion of cycle cost for matrix multiplication with Tensor Cores, we would even have faster GPUs if the global memory latency could be reduced. 
			We can do this by either increasing the clock frequency of the memory (more cycles per second, but also more heat and higher energy requirements) or by increasing the number of elements that can be transferred at any one time (bus width).
	2. Memory Bandwidth
		From the previous section, we have seen that Tensor Cores are very fast. So fast, in fact, that they are idle most of the time as they are waiting for memory to arrive from global memory. For example, during BERT Large training, which uses huge matrices — the larger, the better for Tensor Cores — we have a Tensor Core TFLOPS utilization of about 30%, meaning that 70% of the time, Tensor Cores are idle.
		This means that when comparing two GPUs with Tensor Cores, one of the single best indicators for each GPU’s performance is their memory bandwidth. 
		For example, The A100 GPU has 1,555 GB/s memory bandwidth vs the 900 GB/s of the V100. As such, a basic estimate of speedup of an A100 vs V100 is 1555/900 = 1.73x.
	3. Shared Memory / L1 Cache Size / Registers
		To perform matrix multiplication, we exploit the memory hierarchy of a GPU that goes from slow global memory to fast local shared memory, to lightning-fast registers. However, the faster the memory, the smaller it is. As such, we need to separate the matrix into smaller matrices. We perform matrix multiplication across these smaller tiles in local shared memory that is fast and close to the streaming multiprocessor (SM) — the equivalent of a CPU core. With Tensor Cores, we go a step further: We take each tile and load a part of these tiles into Tensor Cores. A matrix memory tile in shared memory is 10-50x faster than the global GPU memory, whereas the Tensor Cores’ registers are 200x faster than the global GPU memory. You can see TPUs as having very, very, large tiles for each Tensor Core. 
		Shared memory sizes on the following architectures:
			- Volta: 96kb shared memory / 32 kb L1
			- Turing: 64kb shared memory / 32 kb L1
			- Ampere: 164 kb shared memory / 32 kb L1
2. Estimating Ampere Deep Learning Performance
	- Summary
		1. Theoretical estimates based on memory bandwidth and the improved memory hierarchy of Ampere GPUs predict a speedup of 1.78x to 1.87x.
		2. NVIDIA provides accuracy benchmark data of Tesla A100 and V100 GPUs. These data are biased for marketing purposes, but it is possible to build a debiased model of these data.
		3. Debiased benchmark data suggests that the Tesla A100 compared to the V100 is 1.70x faster for NLP and 1.45x faster for computer vision.
	To get an unbiased estimate, we can scale the V100 and A100 results in two ways: (1) account for the differences in batch size, (2) account for the differences in using 1 vs 8 GPUs.
3. Additional Considerations for Ampere / RTX 30 Series
	- Summary
		1. Ampere allows for sparse network training, which accelerates training by a factor of up to 2x.
		2. Sparse network training is still rarely used but will make Ampere future-proof.
		3. Ampere has new low-precision data types, which makes using low-precision much easy, but not necessarily faster than for previous GPUs.
		4. The new fan design is excellent if you have space between GPUs, but it is unclear if multiple GPUs with no space in-between them will be efficiently cooled.
		5. 3-Slot design of the RTX 3090 makes 4x GPU builds problematic. Possible solutions are 2-slot variants or the use of PCIe extenders.
		6. 4x RTX 3090 will need more power than any standard power supply unit on the market can provide right now. 
	- Sparse Network Training
	- Low-precision Computation
		Currently, if you want to have stable backpropagation with **16-bit floating-point numbers (FP16)**, the big problem is that ordinary FP16 data types only support numbers in the range [-65,504, 65,504]. If your gradient slips past this range, your gradients explode into NaN values. To prevent this during FP16 training, we usually perform loss scaling where you multiply the loss by a small number before backpropagating to prevent this gradient explosion. 
		**The Brain Float 16 format (BF16)** uses more bits for the exponent such that the range of possible numbers is the same as for FP32: [-3x10^38, 3x10^38]. BF16 has less precision, that is significant digits, but gradient precision is not that important for learning. So what BF16 does is that you no longer need to do any loss scaling or worry about the gradient blowing up quickly. As such, we should see an increase in training stability by using the BF16 format as a slight loss of precision.
		What this means for you: With BF16 precision, training might be more stable than with FP16 precision while providing the same speedups. With TF32 precision, you get near FP32 stability while giving the speedups close to FP16. The good thing is, to use these data types, you can just replace FP32 with TF32 and FP16 with BF16 — no code changes required!
		Overall, though, these new data types can be seen as lazy data types in the sense that you could have gotten all the benefits with the old data types with some additional programming efforts (proper loss scaling, initialization, normalization, using Apex). As such, these data types do not provide speedups but rather improve ease of use of low precision for training.
	- New Fan Design / Thermal Issues
		1. Thermal
			- water cooling
			- PCIe extenders
		2. Power
			- Power limits
4. GPU Deep Learning Performance, Referred to refernce [2]
5. GPU Recommendations
	- The steps in selecting the best deep learning GPU for you should be:
		1. What do I want to do with the GPU(s): Kaggle competitions, machine learning, learning deep learning, hacking on small projects (GAN-fun or big language models?), doing research in computer vision / natural language processing / other domains, or something else?
		2. How much memory do I need for what I want to do?
		3. Use the Cost/Performance charts from above to figure out which GPU is best for you that fulfills the memory criteria.
		4. Are there additional caveats for the GPU that I chose? For example, if it is an RTX 3090, can I fit it into my computer? Does my power supply unit (PSU) have enough wattage to support my GPU(s)? Will heat dissipation be a problem, or can I somehow cool the GPU effectively?
	- Some guidance
		1. When do I need >= 11 GB of Memory?
		2. When is < 11 GB of Memory Okay?
		3. How can I fit +24GB models into 10GB memory?
			- FP16/BF16 training (apex)
				https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c
			- Gradient checkpointing (only store some of the activations and recompute them in the backward pass)
				https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
			- GPU-to-CPU Memory Swapping (swap layers not needed to the CPU; swap them back in just-in-time for backprop)
				https://arxiv.org/abs/2002.05645v5
			- Model Parallelism (each GPU holds a part of each layer; supported by fairseq)
				https://timdettmers.com/2014/11/09/model-parallelism-deep-learning/
			- Pipeline parallelism (each GPU hols a couple of layers of the network)
			- ZeRO parallelism (each GPU holds partial layers)
				https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
			- 3D parallelism (Model + pipeline + ZeRO)
				https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/
			- CPU Optimizer state (store and update Adam/Momentum on the CPU while the next GPU forward pass is happening)
		If you are not afraid to tinker a bit and implement some of these techniques — which usually means integrating packages that support them with your code — you will be able to fit that 24GB large network on a smaller GPU. With that hacking spirit, the **RTX 3080**, or any GPU with less than 11 GB memory, might be a great GPU for you.
6. Question & Answers & Misconceptions
	- Summary
		- PCIe 4.0 and PCIe lanes do not matter in 2x GPU setups. For 4x GPU setups, they still do not matter much.
		- RTX 3090 and RTX 3080 cooling will be problematic. Use water-cooled cards or PCIe extenders.
		- NVLink is not useful. Only useful for GPU clusters.
		- You can use different types of GPUs in one computer (e.g., GTX 1080 + RTX 2080 + RTX 3090), but you will not be able to parallelize across them efficiently.
		- You will need Infiniband +50Gbit/s networking to parallelize training across more than two machines.
		- AMD CPUs are cheaper than Intel CPUs; Intel CPUs have almost no advantage.
		- Despite heroic software engineering efforts, AMD GPUs + ROCm will probably not be able to compete with NVIDIA due to lacking community and Tensor Core equivalent for at least 1-2 years.
		- Cloud GPUs are useful if you use them for less than 1 year. After that, a desktop is the cheaper solution.
	- When is it better to use the cloud vs a dedicated GPU desktop/server?
		- Rule-of-thumb: If you expect to do deep learning for longer than a year, it is cheaper to get a desktop GPU. Otherwise, cloud instances are preferable unless you have extensive cloud computing skills and want the benefits of scaling the number of GPUs up and down at will.
		
		- For the exact point in time when a cloud GPU is more expensive than a desktop depends highly on the service that you are using, and it is best to do a little math on this yourself. 

		- Below I do an example calculation for an **AWS V100** spot instance with 1x V100 and compare it to the price of a desktop with **a single RTX 3090 (similar performance)**. The desktop with RTX 3090 costs $2,200 (2-GPU barebone + RTX 3090). Additionally, assuming you are in the US, there is an additional $0.12 per kWh for electricity. This compares to $2.14 per hour for the AWS on-demand instance.

		- At 15% utilization per year, the desktop uses: 
			(350 W (GPU) + 100 W (CPU))x0.15 (utilization) * 24 hours * 365 days = 591 kWh per year
			So 591 kWh of electricity per year, that is an additional $71.
		
		- The break-even point for a desktop vs a cloud instance at 15% utilization (you use the cloud instance 15% of time during the day, ), would be about 300 days ($2,311 vs $2,270):
			$2.14/h * 0.15 * 24 hours * 300 days = $2,311,  utilization 3h 36m
		So **if you expect to run deep learning models after 300 days**, it is better to buy a desktop instead of using AWS on-demand instances.
		
		- AWS spot instances are a bit cheaper at about 0.9$ per hour. However, many users on Twitter were telling me that on-demand instances are a nightmare, but that spot instances are hell. AWS itself lists the average frequency of interruptions of V100 GPU spot instances to be above 20%. This means you need a pretty good spot instance management infrastructure to make it worth it to use spot instances. But if you have it, AWS spot instances and similar services are pretty competitive. You need to own and run a desktop for 20 months to run even compared to AWS spot instances. This means if you expect to run deep learning workloads in the next 20 months, a desktop machine will be cheaper (and easier to use).
		
		- You can do similar calculations for any cloud service to make the decision if you go for a cloud service or a desktop.

		- Common utilization rates are the following:
			PhD student personal desktop: < 15%
			PhD student slurm GPU cluster: > 35%
			Company-wide slurm research cluster: > 60%
		In general, utilization rates are lower for professions where thinking about cutting edge ideas is more important than developing practical products. Some areas have low utilization rates (interpretability research), while other areas have much higher rates (machine translation, language modeling). In general, the utilization of personal machines is almost always overestimated. Commonly, most personal systems have a utilization rate between 5-10%. This is why I would highly recommend slurm GPU clusters for research groups and companies instead of individual desktop GPU machines.
7. TL;DR advice
	1. Best GPU overall: RTX 3080 and RTX 3090.
	2. GPUs to avoid (as an individual): Any Tesla card; any Quadro card; any Founders Edition card; Titan RTX, Titan V, Titan XP.
	3. Cost-efficient but expensive: RTX 3080.
	4. Cost-efficient and cheaper:  RTX 3070, RTX 2060 Super
	5. I have little money: Buy used cards. Hierarchy: RTX 2070 ($400), RTX 2060 ($300), GTX 1070 ($220), GTX 1070 Ti ($230), GTX 1650 Super ($190), GTX 980 Ti (6GB $150).
	6. I have almost no money: There are a lot of startups that promo their clouds: Use free cloud credits and switch companies accounts until you can afford a GPU.
	7. I do Kaggle: RTX 3070.
	8. I am a competitive computer vision, pretraining, or machine translation researcher: 4x RTX 3090. Wait until working builds with good cooling, and enough power are confirmed (I will update this blog post).
	9. I am an NLP researcher: If you do not work on machine translation, language modeling, or pretraining of any kind, an RTX 3080 will be sufficient and cost-effective.
	10. I started deep learning, and I am serious about it: Start with an RTX 3070. If you are still serious after 6-9 months, sell your RTX 3070 and buy 4x RTX 3080. Depending on what area you choose next (startup, Kaggle, research, applied deep learning), sell your GPUs, and buy something more appropriate after about three years (next-gen RTX 40s GPUs).
	11. I want to try deep learning, but I am not serious about it: The RTX 2060 Super is excellent but may require a new power supply to be used. If your motherboard has a PCIe x16 slot and you have a power supply with around 300 W, a GTX 1050 Ti is a great option since it will not require any other computer components to work with your desktop computer.
	12. GPU Cluster used for parallel models across less than 128 GPUs: If you are allowed to buy RTX GPUs for your cluster: 66% 8x RTX 3080 and 33% 8x RTX 3090 (only if sufficient cooling is guaranteed/confirmed). If cooling of RTX 3090s is not sufficient buy 33% RTX 6000 GPUs or 8x Tesla A100 instead. If you are not allowed to buy RTX GPUs, I would probably go with 8x A100 Supermicro nodes or 8x RTX 6000 nodes.
	13. GPU Cluster used for parallel models across 128 GPUs: Think about 8x Tesla A100 setups. If you use more than 512 GPUs, you should think about getting a DGX A100 SuperPOD system that fits your scale.

### A Full Hardware Guide to Deep Learning, timedettemers, [2]
1. GPU
	- Research that is hunting state-of-the-art scores: >=11 GB
	- Research that is hunting for interesting architectures: >=8 GB
	- Any other research: 8 GB
	- Kaggle: 4 – 8 GB
	- Startups: 8 GB (but check the specific application area for model sizes)
	- Companies: 8 GB for prototyping, >=11 GB for training
	- Another problem to watch out for, especially if you buy multiple RTX cards is cooling. If you want to stick GPUs into PCIe slots which are next to each other you should make sure that you get GPUs with a blower-style fan. Otherwise you might run into temperature issues and your GPUs will be slower (about 30%) and die faster.
2. RAM
	RAM size does not affect deep learning performance. However, it might hinder you from executing your GPU code comfortably (without swapping to disk). You should have enough RAM to comfortable work with your GPU. This means you should have at least the amount of RAM that matches your biggest GPU. For example, if you have a Titan RTX with 24 GB of memory you should have at least 24 GB of RAM. However, if you have more GPUs you do not necessarily need more RAM.
	The problem with this “match largest GPU memory in RAM” strategy is that you might still fall short of RAM if you are processing large datasets. The best strategy here is to match your GPU and if you feel that you do not have enough RAM just buy some more.
	A different strategy is influenced by psychology: Psychology tells us that concentration is a resource that is depleted over time. RAM is one of the few hardware pieces that allows you to conserve your concentration resource for more difficult programming problems. Rather than spending lots of time on circumnavigating RAM bottlenecks, you can invest your concentration on more pressing matters if you have more RAM.  **With a lot of RAM you can avoid those bottlenecks, save time and increase productivity on more pressing problems.** Especially in Kaggle competitions, I found additional RAM very useful for feature engineering. So if you have the money and do a lot of pre-processing then additional RAM might be a good choice. So with this strategy, you want to have more, cheap RAM now rather than later.
3. CPU
	The main mistake that people make is that people pay too much attention to PCIe lanes of a CPU. You should not care much about PCIe lanes. Instead, just look up if your CPU and motherboard combination supports the number of GPUs that you want to run. The second most common mistake is to get a CPU which is too powerful.
	1. CPU and PCI-Express
		When you select CPU PCIe lanes and motherboard PCIe lanes make sure that you select a combination which supports the desired number of GPUs. If you buy a motherboard that supports 2 GPUs, and you want to have 2 GPUs eventually, make sure that you buy a CPU that supports 2 GPUs, but do not necessarily look at PCIe lanes.
	2. PCIe Lanes and Multi-GPU Parallelism
		Since almost nobody runs a system with more than 4 GPUs as a rule of thumb: Do not spend extra money to get more PCIe lanes per GPU — it does not matter!
	3. Needed CPU Cores
		To be able to make a wise choice for the CPU we first need to understand the CPU and how it relates to deep learning. What does the CPU do for deep learning? The CPU does little computation when you run your deep nets on a GPU. Mostly it (1) initiates GPU function calls, (2) executes CPU functions.
		By far the most useful application for your CPU is data preprocessing. There are two different common data processing strategies which have different CPU needs.
		
		- The first strategy is preprocessing while you train:
			- Loop
				1. Load mini-batch
				2. Preprocess mini-batch
				3. Train on mini-batch
		- The second strategy is preprocessing before any training:
			1. Preprocess data
			2. Loop
				1. Load preprocessed mini-batch
				2. Train on mini-batch
		For the first strategy, a good CPU with many cores can boost performance significantly. For the second strategy, you do not need a very good CPU. For the first strategy, I recommend a minimum of 4 threads per GPU — that is usually two cores per GPU. I have not done hard tests for this, but you should gain about 0-5% additional performance per additional core/GPU.
		For the second strategy, **I recommend a minimum of 2 threads per GPU** — that is usually one core per GPU. You will not see significant gains in performance when you have more cores if you are using the second strategy.
	4. Needed CPU Clock Rate (Frequency)
		When people think about fast CPUs they usually first think about the clock rate.  4GHz is better than 3.5GHz, or is it? This is generally true for comparing processors with the same architecture, e.g. “Ivy Bridge”, but it does not compare well between processors. Also, it is not always the best measure of performance.
		In the case of deep learning there is very little computation to be done by the CPU: Increase a few variables here, evaluate some Boolean expression there, make some function calls on the GPU or within the program – all these depend on the CPU core clock rate.
		While this reasoning seems sensible, there is the fact that the CPU has 100% usage when I run deep learning programs, so what is the issue here? I did some CPU core rate underclocking experiments to find out.				
		Note that these experiments are on a hardware that is dated, however, these results should still be the same for modern CPUs/GPUs.
4. HDD/SSD
5. Power Supply Unit(PSU)
6. CPU and GPU Cooling
	1. Air Cooling GPUs
		**Air cooling is safe and solid for a single GPU or if you have multiple GPUs with space between them (2 GPUs in a 3-4 GPU case)**. However, one of the biggest mistakes can be made when you try to cool 3-4 GPUs and you need to think carefully about your options in this case.
		Modern GPUs will increase their speed – and thus power consumption – up to their maximum when they run an algorithm, but as soon as the GPU hits a temperature barrier – often 80 °C – the GPU will decrease the speed so that the temperature threshold is not breached. This enables the best performance while keeping your GPU safe from overheating.
		However, typical pre-programmed schedules for fan speeds are badly designed for deep learning programs, so that this temperature threshold is reached within seconds after starting a deep learning program. The result is a decreased performance (0-10%) which can be significant for multiple GPUs (10-25%) where the GPU heat up each other.
		Since NVIDIA GPUs are first and foremost gaming GPUs, they are optimized for Windows. You can change the fan schedule with a few clicks in Windows, but not so in Linux, and as most deep learning libraries are written for Linux this is a problem.
		The only option under Linux is to use to set a configuration for your Xorg server (Ubuntu) where you set the option “coolbits”. This works very well for a single GPU, but if you have multiple GPUs where some of them are headless, i.e. they have no monitor attached to them, you have to emulate a monitor which is hard and hacky. I tried it for a long time and had frustrating hours with a live boot CD to recover my graphics settings – I could never get it running properly on headless GPUs.
		The most important point of consideration if you run 3-4 GPUs on air cooling is to pay attention to the fan design. The “blower” fan design pushes the air out to the back of the case so that fresh, cooler air is pushed into the GPU. Non-blower fans suck in air in the vincity of the GPU and cool the GPU. However, if you have multiple GPUs next to each other then there is no cool air around and GPUs with non-blower fans will heat up more and more until they throttle themselves down to reach cooler temperatures. **Avoid non-blower fans in 3-4 GPU setups at all costs.**
	2. Water Cooling GPUs For Multiple GPUs
		Another, more costly, and craftier option is to use water cooling. I do not recommend water cooling if you have a single GPU or if you have space between your two GPUs (2 GPUs in 3-4 GPU board). However, water cooling makes sure that even the beefiest GPU stay cool in a 4 GPU setup which is not possible when you cool with air. Another advantage of water cooling is that it operates much more silently, which is a big plus if you run multiple GPUs in an area where other people work. Water cooling will cost you about $100 for each GPU and some additional upfront costs (something like $50). Water cooling will also require some additional effort to assemble your computer, but there are many detailed guides on that and it should only require a few more hours of time in total. Maintenance should not be that complicated or effortful.
	3. Big case? Not useful
	4. Conclusion Cooling
		So in the end it is simple: For 1 GPU air cooling is best. For multiple GPUs, you should get blower-style air cooling and accept a tiny performance penalty (10-15%), or you pay extra for water cooling which is also more difficult to setup correctly and you have no performance penalty. Air and water cooling are all reasonable choices in certain situations. I would however recommend air cooling for simplicity in general — get a blower-style GPU if you run multiple GPUs. If you want to user water cooling try to find all-in-one (AIO) water cooling solutions for GPUs.
7. Motherboard
	Your motherboard should have **enough PCIe ports to support the number of GPUs you want to run (usually limited to four GPUs, even if you have more PCIe slots)**
	Remember that most GPUs have a width of two PCIe slots, so buy a motherboard that has enough space between PCIe slots if you intend to use multiple GPUs. Make sure your motherboard not only has the PCIe slots, but actually supports the GPU setup that you want to run. You can usually find information in this if you search your motherboard of choice on newegg and look at PCIe section on the specification page.
8. Computer Case
9. Monitor

## Refernece
1. https://www.romaglushko.com/blog/how-i-built-my-ml-workstation/
2. https://timdettmers.com/
3. PCPartPicker: https://pcpartpicker.com/user/tim_dettmers/saved/#view=mZ2rD3
4. TPU vs GPU: https://timdettmers.com/2018/10/17/tpus-vs-gpus-for-transformers-bert/
